#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Bollinger Band squeeze breakout with 1w EMA200 trend filter
# Long when price breaks above upper BB(20,2) AND 1w close > 1w EMA200 (bullish regime) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below lower BB(20,2) AND 1w close < 1w EMA200 (bearish regime) AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back through 20-period SMA (mean reversion to middle band)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Bollinger Band squeeze identifies low volatility periods primed for breakout
# 1w EMA200 regime filter ensures we trade with the dominant weekly trend
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "1d_1wBB_Squeeze_Breakout_1wEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Bollinger Bands and EMA200
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need at least 200 completed weekly bars for EMA200
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Bollinger Bands (20,2)
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + (2.0 * std_20_1w)
    lower_bb_1w = sma_20_1w - (2.0 * std_20_1w)
    
    # Align 1w Bollinger Bands to 1d timeframe (wait for completed 1w bar)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB AND 1w close > 1w EMA200 (bullish) AND volume spike
            if (close[i] > upper_bb_aligned[i] and 
                close_1w[-1] > ema_200_1w_aligned[i] and  # Use latest completed 1w close for regime
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB AND 1w close < 1w EMA200 (bearish) AND volume spike
            elif (close[i] < lower_bb_aligned[i] and 
                  close_1w[-1] < ema_200_1w_aligned[i] and  # Use latest completed 1w close for regime
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 20-period SMA (mean reversion)
            if close[i] < sma_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 20-period SMA (mean reversion)
            if close[i] > sma_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals