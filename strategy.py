#!/usr/bin/env python3
name = "6h_Liquidity_Sweep_Reversal_1wTrend"
timeframe = "6h"
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
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    trend_up = close > ema_100_1w_aligned
    trend_down = close < ema_100_1w_aligned
    
    # Rolling 50-period highs/lows for liquidity levels
    high_50 = pd.Series(high).rolling(window=50, min_periods=50).max().values
    low_50 = pd.Series(low).rolling(window=50, min_periods=50).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day (4*6h) to reduce trade frequency
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_50[i]) or 
            np.isnan(low_50[i]) or 
            np.isnan(ema_100_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price sweeps below 50-period low then closes above it with volume in 1w uptrend
            if (low[i] < low_50[i] and 
                close[i] > low_50[i] and 
                trending_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price sweeps above 50-period high then closes below it with volume in 1w downtrend
            elif (high[i] > high_50[i] and 
                  close[i] < high_50[i] and 
                  trending_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price closes below 50-period low or 1w trend changes to down
            if close[i] < low_50[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above 50-period high or 1w trend changes to up
            if close[i] > high_50[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, liquidity sweeps (stop hunts) followed by reversals with volume confirmation and 1-week EMA100 trend filter captures smart money reversals. 
# Liquidity sweeps occur when price briefly breaks recent highs/lows to trigger stops before reversing - common in ranging markets. 
# The 1-week trend filter ensures we only take reversals in the direction of the higher timeframe trend, avoiding counter-trend traps. 
# Volume confirmation ensures institutional participation in the reversal. 
# This strategy works in both bull and bear markets: in bull markets we take long reversals from liquidity sweeps below support, 
# in bear markets we take short reversals from liquidity sweeps above resistance. 
# Uses 50-period lookback for liquidity levels (sufficiently long to identify meaningful liquidity pools) and 4-bar cooldown to limit trades to ~15-35/year. 
# Position size of 0.25 balances risk and return while minimizing fee churn from excessive trading.