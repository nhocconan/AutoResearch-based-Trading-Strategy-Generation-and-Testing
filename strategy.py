#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 6h Supertrend (ATR=10, mult=3.0) with 1w ADX regime filter and volume confirmation.
# Long when Supertrend gives buy signal, 1w ADX > 25 (trending market), and 6h volume > 1.5x 20-period median volume.
# Short when Supertrend gives sell signal, 1w ADX > 25, and same volume condition.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Supertrend avoids whipsaws in ranging markets, ADX ensures we only trade in trending conditions (works in both bull/bear trends),
# volume confirmation adds conviction to breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once before loop for Supertrend and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # === 6h Indicators: Supertrend (ATR=10, mult=3.0) and volume median ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    vol_6h = df_6h['volume'].values
    
    # Calculate ATR(10)
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend
    hl2 = (high_6h + low_6h) / 2
    upperband = hl2 + (3.0 * atr_10)
    lowerband = hl2 - (3.0 * atr_10)
    
    supertrend = np.zeros_like(close_6h)
    direction = np.ones_like(close_6h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_6h)):
        if close_6h[i] > upperband[i-1]:
            direction[i] = 1
        elif close_6h[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        supertrend[i] = upperband[i] if direction[i] == 1 else lowerband[i]
    
    # Supertrend signal: 1 for buy (close > supertrend), -1 for sell (close < supertrend)
    supertrend_signal = np.where(close_6h > supertrend, 1, -1)
    
    # Calculate 6h volume median (20-period)
    vol_median_20 = pd.Series(vol_6h).rolling(window=20, min_periods=20).median().values
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: ADX(14) for trend strength ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # True Range
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(np.maximum(tr1_w, tr2_w), tr3_w)
    tr_w[0] = tr1_w[0]  # First TR is just high-low
    
    # Smoothed values
    atr_1w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (6h)
    supertrend_signal_aligned = align_htf_to_ltf(prices, df_6h, supertrend_signal)
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    vol_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_6h)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 10, 14, 14)  # volume median(20), ATR(10), ADX components
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_signal_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_6h_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        st_signal = supertrend_signal_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_6h = vol_6h_aligned[i]
        adx_1w = adx_1w_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when Supertrend flips to sell signal
            if st_signal == -1:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when Supertrend flips to buy signal
            if st_signal == 1:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 6h volume > 1.5x median volume
            volume_spike = vol_6h > (vol_median * 1.5)
            # Regime filter: 1w ADX > 25 (trending market)
            trending_market = adx_1w > 25
            
            # LONG CONDITIONS
            # Supertrend buy signal, trending market, and volume spike
            if st_signal == 1 and trending_market and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Supertrend sell signal, trending market, and volume spike
            elif st_signal == -1 and trending_market and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_Supertrend_ADX1w_VolumeSpike1.5x_v1"
timeframe = "6h"
leverage = 1.0