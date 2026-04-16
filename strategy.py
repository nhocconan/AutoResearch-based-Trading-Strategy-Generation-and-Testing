#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h CCI(20) combined with 1d ADX(14) trend filter and volume confirmation.
# CCI > +100 indicates strong uptrend, CCI < -100 indicates strong downtrend.
# In trending markets (ADX > 25): follow CCI signals (long >+100, short <-100).
# In ranging markets (ADX < 20): mean reversion at CCI extremes (>+200 long, <-200 short).
# Volume confirmation (>1.5x average) filters weak signals. Position size 0.25.
# Designed to work in bull (trend following) and bear (mean reversion in ranges).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for ADX trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h CCI(20) ===
    tp_12h = (high_12h + low_12h + close_12h) / 3.0
    sma_tp = pd.Series(tp_12h).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_12h).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp_12h - sma_tp) / (0.015 * mad)
    cci = np.where(mad == 0, 0, cci)
    cci_cci = align_htf_to_ltf(prices, df_12h, cci)
    
    # === 1d ADX(14) for trend filter ===
    # Calculate +DM, -DM, TR
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0]) * -1  # inverted for calculation
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    adx_cci = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(cci_cci[i]) or 
            np.isnan(adx_cci[i]) or
            np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        cci_val = cci_cci[i]
        adx_val = adx_cci[i]
        vol_ratio = vol_ratio_12h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when CCI returns to neutral or trend weakens
            if cci_val < 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when CCI returns to neutral or trend weakens
            if cci_val > 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            if adx_val > 25:  # Trending market
                # Follow CCI trend
                if cci_val > 100 and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                elif cci_val < -100 and vol_ratio > 1.5:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
            else:  # Ranging market (ADX < 25)
                # Mean reversion at extreme CCI levels
                if cci_val < -200 and vol_ratio > 1.5:  # Deep oversold
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                elif cci_val > 200 and vol_ratio > 1.5:  # Deep overbought
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

name = "12h_CCI_ADX_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0