#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1w trend filter + volume confirmation
# Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Camarilla R3/S3 levels provide institutional pivot points with proven effectiveness
# 1w EMA50 trend filter ensures trades align with higher timeframe momentum
# Volume spike (2x 20-period average) confirms participation
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels (prior completed 12h bar's range)
    daily_high = pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values
    daily_low = pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values
    daily_close = pd.Series(close).rolling(window=2, min_periods=2).last().shift(1).values
    
    # Camarilla pivot = (H+L+C)/3
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # R3 = pivot + (H-L) * 1.1/4
    r3 = daily_pivot + daily_range * 1.1 / 4.0
    # S3 = pivot - (H-L) * 1.1/4
    s3 = daily_pivot - daily_range * 1.1 / 4.0
    
    # Calculate 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align HTF indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, None, r3)  # R3 already calculated on 12h data
    s3_aligned = align_htf_to_ltf(prices, None, s3)  # S3 already calculated on 12h data
    daily_pivot_aligned = align_htf_to_ltf(prices, None, daily_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1w EMA50 (bullish trend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below S3 AND price < 1w EMA50 (bearish trend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 OR below 1w EMA50 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 OR above 1w EMA50 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals