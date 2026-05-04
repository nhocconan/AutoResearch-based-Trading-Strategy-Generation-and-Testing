#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA21 trend + volume confirmation
# Donchian breakout captures momentum; 1w HMA21 ensures alignment with weekly trend.
# Volume spike (1.5x 20-period EMA) confirms participation and reduces false breakouts.
# Designed for 1d timeframe to target 7-25 trades/year (30-100 total over 4 years).
# Works in bull markets via breakouts with trend, and in bear markets via short breakouts against trend.
# Discrete sizing (0.25) minimizes fee churn.

name = "1d_Donchian20_1wHMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators (same timeframe, but we need it for alignment)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) bands
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w HMA21 for trend filter
    close_1w = df_1w['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_1w = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Donchian breakout with 1w HMA21 trend filter
        # Long: break above upper band + volume spike + price above 1w HMA21 (uptrend)
        # Short: break below lower band + volume spike + price below 1w HMA21 (downtrend)
        if position == 0:
            if (close[i] > highest_high[i] and volume_spike and 
                close[i] > hma_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lowest_low[i] and volume_spike and 
                  close[i] < hma_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below lower band (mean reversion) OR below 1w HMA21 (trend change)
            if close[i] < lowest_low[i] or close[i] < hma_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above upper band (mean reversion) OR above 1w HMA21 (trend change)
            if close[i] > highest_high[i] or close[i] > hma_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals