# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout on 1d with 1w trend filter and volume confirmation
# Works in bull/bear: Trend filter adapts to long/short conditions, volume confirms breakout strength
# Target: 30-100 trades over 4 years (7-25/year) with 1d timeframe
# Uses 1w trend filter (HMA) to avoid counter-trend trades in choppy markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels (primary entry/exit)
    R1 = prev_close + 0.25 * (prev_high - prev_low)
    S1 = prev_close - 0.25 * (prev_high - prev_low)
    
    # Get 1w data for trend filter (HMA 21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate HMA 21 on 1w close for trend filter
    close_1w = df_1w['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    # Pad arrays for WMA calculation
    wma_half = np.full_like(close_1w, np.nan)
    wma_full = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= half_len:
        wma_half[half_len-1:] = wma(close_1w, half_len)
    if len(close_1w) >= 21:
        wma_full[20:] = wma(close_1w, 21)
    
    # HMA calculation
    hma_21 = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 21 and len(close_1w) >= half_len:
        # 2*WMA(n/2) - WMA(n)
        diff = 2 * wma_half - wma_full
        # WMA of the difference with sqrt(n) window
        if len(diff) >= sqrt_len:
            wma_diff = wma(diff, sqrt_len)
            hma_21[sqrt_len-1:] = wma_diff
    
    # Align indicators to 1d timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Volume average for spike detection (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        hma_1w = hma_21_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Close > R1 AND price > 1w HMA21 (uptrend) AND volume > 2.0x average
            if close[i] > r1 and close[i] > hma_1w and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S1 AND price < 1w HMA21 (downtrend) AND volume > 2.0x average
            elif close[i] < s1 and close[i] < hma_1w and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S1 OR trend reverses (price < 1w HMA21)
            if close[i] < s1 or close[i] < hma_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R1 OR trend reverses (price > 1w HMA21)
            if close[i] > r1 or close[i] > hma_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals