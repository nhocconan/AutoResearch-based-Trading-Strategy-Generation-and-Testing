#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) AND 1d close > SMA50 (uptrend) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below lower BB(20,2) AND 1d close < SMA50 (downtrend) AND volume > 1.5 * avg_volume(20)
# Exit when price returns to middle BB (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Bollinger squeeze breakouts capture volatility expansion after low volatility periods
# 1d SMA50 trend filter ensures we trade with the dominant daily trend
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "4h_1dBB_Squeeze_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Bollinger Bands and SMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for SMA50
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2.0 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2.0 * std_20_1d)
    middle_bb_1d = sma_20_1d
    
    # Align 1d Bollinger Bands to 4h timeframe (wait for completed 1d bar)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb_1d)
    
    # Calculate 1d SMA50 for trend filter
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
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
            np.isnan(middle_bb_aligned[i]) or np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB, 1d close > SMA50 (uptrend), volume spike, in session
            if (close[i] > upper_bb_aligned[i] and 
                close_1d[-1] > sma_50_1d_aligned[i] and  # Use latest completed 1d close
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB, 1d close < SMA50 (downtrend), volume spike, in session
            elif (close[i] < lower_bb_aligned[i] and 
                  close_1d[-1] < sma_50_1d_aligned[i] and  # Use latest completed 1d close
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB (mean reversion)
            if close[i] < middle_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB (mean reversion)
            if close[i] > middle_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals