#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and daily for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly trend filter: EMA50
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Camarilla R3 and S3 levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    r3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    s3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    
    # Align daily levels to 12h timeframe (with 1-day delay for completed bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 2x 24-period average (2 days for 12h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # 2 days for 12h timeframe
    
    start_idx = max(100, 24, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        trend_up = close > ema_50_1w_aligned[i]
        trend_down = close < ema_50_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R3 in uptrend with strong volume
            if (close[i] > r3_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.30
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S3 in downtrend with strong volume
            elif (close[i] < s3_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.30
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters Camarilla body (between R3 and S3) or trend change
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price re-enters Camarilla body or trend change
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Weekly EMA50 trend filter with daily Camarilla R3/S3 breakouts and volume confirmation
# on 12h timeframe targets 50-150 trades over 4 years. Weekly trend ensures alignment with major market
# direction, reducing false breakouts. Volume >2x 2-day average confirms institutional participation.
# Position size 0.30 balances profit potential with drawdown control. Works in bull (breakouts above R3)
# and bear (breakdowns below S3) by trading with weekly trend. Cooldown of 4 bars (2 days) prevents
# overtrading and reduces fee drag. Target: 12-37 trades/year per symbol.