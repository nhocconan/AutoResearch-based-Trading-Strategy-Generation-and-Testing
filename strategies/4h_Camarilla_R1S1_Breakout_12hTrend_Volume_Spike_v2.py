#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume_Spike_v2"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Camarilla levels from previous 1d bar
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    R1 = prev_close + range_ * 1.1 / 12
    R2 = prev_close + range_ * 1.1 / 6
    R3 = prev_close + range_ * 1.1 / 4
    S1 = prev_close - range_ * 1.1 / 12
    S2 = prev_close - range_ * 1.1 / 6
    S3 = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    # Volume filter: current volume > 2.0x 24-period average (4 days for 4h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1 day for 4h to reduce trades
    
    start_idx = max(100, 24, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 12h trend direction
        trend_up = close > ema_50_12h_aligned[i]
        trend_down = close < ema_50_12h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above R1 with volume in uptrend
            if (close[i] > R1_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S1 with volume in downtrend
            elif (close[i] < S1_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below S1 or trend changes
            if close[i] < S1_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above R1 or trend changes
            if close[i] > R1_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike.
# Long when price breaks above R1 in uptrend with volume confirmation.
# Short when price breaks below S1 in downtrend with volume confirmation.
# Uses 4h timeframe for optimal balance of signal quality and trade frequency.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).