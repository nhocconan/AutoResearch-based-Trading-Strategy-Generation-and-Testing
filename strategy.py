#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h CCI + ADX Trend Filter with Volume Confirmation
# CCI(20) > 100 signals overbought/strong uptrend; CCI(20) < -100 signals oversold/strong downtrend.
# ADX(14) > 25 confirms trending market. Volume > 1.3x average confirms momentum.
# In trending markets (ADX > 25): go long when CCI crosses above -100 from below (end of pullback),
# go short when CCI crosses below 100 from above (end of rally).
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (higher timeframe for ADX trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 6h CCI(20) ===
    typical_price_6h = (high_6h + low_6h + close_6h) / 3.0
    tp_mean = pd.Series(typical_price_6h).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price_6h).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    cci = np.where(tp_mad != 0, (typical_price_6h - tp_mean) / (0.015 * tp_mad), 0.0)
    
    # === 1d ADX(14) for trend filter ===
    # Calculate True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = abs(pd.Series(high_1d).diff())
    tr3 = abs(pd.Series(low_1d).diff())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(cci[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        cci_val = cci[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_6h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when CCI falls below -100 or trend weakens
            if cci_val < -100 or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when CCI rises above 100 or trend weakens
            if cci_val > 100 or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending market (ADX > 25) and volume confirmation
            if adx_val > 25 and vol_ratio > 1.3:
                # Buy when CCI crosses above -100 from below (end of pullback in uptrend)
                if cci_val > -100 and cci[i-1] <= -100:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Sell when CCI crosses below 100 from above (end of rally in downtrend)
                elif cci_val < 100 and cci[i-1] >= 100:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_CCI_ADX_VolumeTrendFilter_v1"
timeframe = "6h"
leverage = 1.0