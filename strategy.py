#/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 levels from previous 4h bar (not day)
    high_prev = df_4h['high'].shift(1).values
    low_prev = df_4h['low'].shift(1).values
    close_prev = df_4h['close'].shift(1).values
    
    r3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    s3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    
    # Align 4h levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # 4h trend filter: EMA34
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d volume filter: current volume > 2.0x 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma_20_1d
    vol_filter_1d = vol_ratio > 2.0
    vol_filter_1h = align_htf_to_ltf(prices, df_1d, vol_filter_1d.astype(float)) > 0.5
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0 and session_filter[i]:
            # Long: Break above R3 in uptrend with strong daily volume
            if (close[i] > r3_aligned[i] and 
                close > ema_34_4h_aligned[i] and 
                vol_filter_1h[i]):
                signals[i] = 0.20
                position = 1
            # Short: Break below S3 in downtrend with strong daily volume
            elif (close[i] < s3_aligned[i] and 
                  close < ema_34_4h_aligned[i] and 
                  vol_filter_1h[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Price re-enters Camarilla body or trend change
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or close < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price re-enters Camarilla body or trend change
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or close > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Using 4h Camarilla R3/S3 breakouts with 4h EMA34 trend filter and 1d volume confirmation
# on 1h timeframe will yield 15-35 trades/year. The 4h trend ensures we trade with the higher timeframe
# momentum, while 1d volume filter confirms institutional participation. Session filter (08-20 UTC)
# reduces noise from low-liquidity hours. Position size 0.20 manages drawdown. Works in bull (breakouts above R3)
# and bear (breakdowns below S3) by following 4h trend. Target: 60-140 total trades over 4 years.