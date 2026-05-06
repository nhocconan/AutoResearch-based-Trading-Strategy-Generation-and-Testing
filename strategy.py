#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Bollinger Band squeeze breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) on 4h, 1d EMA34 > EMA34 previous 5 bars (strong uptrend), and volume > 1.5 * avg_volume(20)
# Short when price breaks below lower BB(20,2) on 4h, 1d EMA34 < EMA34 previous 5 bars (strong downtrend), and volume > 1.5 * avg_volume(20)
# Exit when price crosses back through 20-period SMA on 4h (mean reversion to middle band)
# Uses discrete sizing 0.25 to balance return and risk while minimizing fee churn
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Bollinger Band squeeze identifies low volatility periods primed for explosive breakouts
# 1d EMA34 trend filter ensures we trade with the dominant daily trend with confirmation
# Volume confirmation (1.5x) validates breakout strength while limiting false signals
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "4h_1dBB_Squeeze_Breakout_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Bollinger Bands on 4h: SMA(20) ± 2*STD(20)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    sma_20_4h = sma_20  # 20-period SMA for exit condition
    
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB, 1d EMA34 trending up (current > 5 bars ago), volume spike, in session
            if (close[i] > upper_bb[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-5] and  # EMA34 up trend confirmation
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB, 1d EMA34 trending down (current < 5 bars ago), volume spike, in session
            elif (close[i] < lower_bb[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-5] and  # EMA34 down trend confirmation
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 20-period SMA (mean reversion)
            if close[i] < sma_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 20-period SMA (mean reversion)
            if close[i] > sma_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals