#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot (S1/R1) breakout with 12h volume confirmation and ADX trend filter.
# In trending markets (ADX > 25): buy breakouts above R1, sell breakdowns below S1.
# Works in bull (breakouts continue) and bear (breakdowns continue) markets.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 12h data (higher timeframe for volume confirmation) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (for Camarilla pivot calculation) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_12h = df_1d['low'].values  # Note: variable name reused for clarity in calc
    close_1d = df_1d['close'].values
    
    # === Camarilla Pivot Levels (S1, R1) from previous 1d bar ===
    # Typical Price = (H + L + C) / 3
    typical_price = (high_1d + low_12h + close_1d) / 3.0
    # Pivot Point = Typical Price
    pp = typical_price
    # R1 = C + (H - L) * 1.1 / 12
    r1 = close_1d + (high_1d - low_12h) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = close_1d - (high_1d - low_12h) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === 12h ADX for trend filter ===
    # Calculate True Range
    tr1 = pd.Series(high_12h).diff()
    tr2 = abs(pd.Series(high_12h).diff())
    tr3 = abs(pd.Series(low_12h).diff())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(high_12h).diff()
    down_move = -pd.Series(low_12h).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        adx_val = adx_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price falls below pivot point or trend weakens
            if price < pp_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above pivot point or trend weakens
            if price > pp_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending market (ADX > 25) and volume confirmation
            if adx_val > 25 and vol_ratio > 1.5:
                # Buy when price breaks above R1 with volume
                if price > r1_val and close[i-1] <= r1_val:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Sell when price breaks below S1 with volume
                elif price < s1_val and close[i-1] >= s1_val:
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

name = "4h_Camarilla_S1R1_Breakout_Volume_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0