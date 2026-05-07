#!/usr/bin/env python3

name = "6h_PostBearBreakout_1dTrend_Confirmation"
timeframe = "6h"
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
    
    # Get 1d and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for long-term trend
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate weekly close for trend
    close_1w = df_1w['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
    
    # Align HTF data
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume filter: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(close_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d and weekly trend
        price_above_ema200 = close[i] > ema_200_1d_aligned[i]
        price_below_ema200 = close[i] < ema_200_1d_aligned[i]
        weekly_uptrend = close_1w_aligned[i] > close_1w_aligned[i-1] if i > 0 else False
        weekly_downtrend = close_1w_aligned[i] < close_1w_aligned[i-1] if i > 0 else False
        
        # Entry conditions
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above upper Donchian in 1d uptrend with volume
            if (close[i] > upper[i] and 
                price_above_ema200 and 
                weekly_uptrend and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: break below lower Donchian in 1d downtrend with volume
            elif (close[i] < lower[i] and 
                  price_below_ema200 and 
                  weekly_downtrend and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: break below lower Donchian or trend change
            if (close[i] < lower[i]) or not price_above_ema200:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: break above upper Donchian or trend change
            if (close[i] > upper[i]) or not price_below_ema200:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: After bear markets, strong breakouts with alignment of 6h price action,
# 1d trend (EMA200), and weekly momentum capture the start of new trends.
# Donchian breakouts provide objective entry/exit levels. Weekly trend filter ensures
# we only trade in the direction of the higher timeframe momentum. Volume confirmation
# avoids false breakouts. Cooldown periods prevent overtrading. Works in bull markets
# by capturing uptrend continuations and in bear markets by catching breakdowns.
# Target: 25-40 trades/year. Position size 0.25 manages drawdown during choppy periods.