#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.0833)
    s1 = pivot - (range_1d * 1.0833)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d data to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume filter: current volume > 1.8x 30-period average (4h)
    vol_ma_30 = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_30[i] = np.mean(volume[i-30:i])
    vol_filter = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~2 days for 4h to reduce trades
    
    start_idx = max(60, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above R1 with volume in uptrend
            if (close[i] > r1_aligned[i] and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.30
                position = 1
                bars_since_last_trade = 0
            # Short: break below S1 with volume in downtrend
            elif (close[i] < s1_aligned[i] and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.30
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price back below S1 or trend change
            if (close[i] < s1_aligned[i]) or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R1 or trend change
            if (close[i] > r1_aligned[i]) or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla R1/S1 levels act as key intraday support/resistance. Breaks above R1 with volume in 1d uptrend signal bullish continuation; breaks below S1 with volume in 1d downtrend signal bearish continuation. The 1d EMA34 filter ensures trades align with higher-timeframe trend, reducing false breakouts. Volume confirmation ensures participation. Cooldown periods prevent overtrading. Works in bull markets by buying R1 breaks in uptrends and in bear markets by selling S1 breaks in downtrends. Target: 20-40 trades/year. Uses discrete position sizing (0.30) to minimize fee churn. 4h timeframe balances signal quality and trade frequency. Camarilla levels provide structured entry points while EMA34 establishes trend direction. Volume ensures participation. This avoids overtrading by requiring multiple confirmations and cooldown.