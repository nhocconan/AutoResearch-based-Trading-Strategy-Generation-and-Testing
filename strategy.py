#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout with 1w EMA Trend Filter and Volume Confirmation
# - Long when price breaks above 20-day high AND weekly EMA200 > weekly EMA50 (bullish)
# - Short when price breaks below 20-day low AND weekly EMA200 < weekly EMA50 (bearish)
# - Volume must be > 1.5x 20-day average volume for confirmation
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 and EMA50
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-day Donchian channels and average volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema200_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        ema200 = ema200_1w_aligned[i]
        ema50 = ema50_1w_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_avg = avg_volume[i]
        
        if position == 0:
            # Long entry: price breaks above 20-day high AND weekly EMA200 > EMA50 (bullish) AND volume confirmation
            if price > upper and ema200 > ema50 and vol > 1.5 * vol_avg:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-day low AND weekly EMA200 < EMA50 (bearish) AND volume confirmation
            elif price < lower and ema200 < ema50 and vol > 1.5 * vol_avg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low or weekly EMA turns bearish
            if price < lower or ema200 < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high or weekly EMA turns bullish
            if price > upper or ema200 > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_20_1wEMA_TrendFilter"
timeframe = "1d"
leverage = 1.0