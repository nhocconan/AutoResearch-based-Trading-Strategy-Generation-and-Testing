#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Bollinger Band squeeze breakout with 1w EMA50 trend filter and volume confirmation
# Long when 1d close breaks above upper Bollinger Band (20,2) AND 1w EMA50 > EMA50 previous (uptrend) AND volume > 2.0 * avg_volume(20)
# Short when 1d close breaks below lower Bollinger Band (20,2) AND 1w EMA50 < EMA50 previous (downtrend) AND volume > 2.0 * avg_volume(20)
# Exit when 1d close crosses back inside the Bollinger Bands
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Bollinger Band squeeze indicates low volatility and impending breakout
# 1w EMA50 trend filter ensures we trade with the dominant weekly trend
# Volume spike confirmation (2.0x) validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "1d_1wBBSqueeze_Breakout_1wEMA50_Trend_VolumeConfirm"
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
    
    # Get 1w data ONCE before loop for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need at least 50 completed weekly bars for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w Bollinger Bands (20,2)
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + (2.0 * std_20_1w)
    lower_bb_1w = sma_20_1w - (2.0 * std_20_1w)
    
    # Align 1w Bollinger Bands to 1d timeframe (wait for completed 1w bar)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close breaks above upper Bollinger Band, 1w EMA50 uptrend, volume spike, in session
            if (close[i] > upper_bb_aligned[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below lower Bollinger Band, 1w EMA50 downtrend, volume spike, in session
            elif (close[i] < lower_bb_aligned[i] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close crosses back inside Bollinger Bands (below upper band)
            if close[i] < upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close crosses back inside Bollinger Bands (above lower band)
            if close[i] > lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals