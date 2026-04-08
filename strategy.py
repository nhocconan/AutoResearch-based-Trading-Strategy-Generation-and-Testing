#!/usr/bin/env python3
# 12h_1d_adx_trend_follow_v1
# Hypothesis: 12-hour ADX trend following with 1-day trend filter. Uses ADX(14) > 25 to identify trending markets,
# then enters long when +DI > -DI and price above 20-period EMA, short when -DI > +DI and price below 20-period EMA.
# The 1-day EMA(50) acts as a higher timeframe trend filter to avoid counter-trend trades.
# Designed to generate ~20-30 trades/year to minimize fee decay while capturing sustained trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_adx_trend_follow_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX components on 12h data
    period_adx = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR and DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) < period:
            return smoothed
        # First value is simple average
        smoothed[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + data[i]
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + data[i]
        return smoothed
    
    tr_smoothed = wilder_smooth(tr, period_adx)
    plus_dm_smoothed = wilder_smooth(plus_dm, period_adx)
    minus_dm_smoothed = wilder_smooth(minus_dm, period_adx)
    
    # DI and ADX
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period_adx)
    
    # 20-period EMA for entry filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1-day for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: ADX < 20 (trend weak) or EMA cross down or DI cross down
            if adx[i] < 20 or ema_20[i] < close[i] or plus_di[i] < minus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: ADX < 20 (trend weak) or EMA cross up or DI cross up
            if adx[i] < 20 or ema_20[i] > close[i] or minus_di[i] < plus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: ADX > 25 (strong trend) and DI crossover with EMA filter
            # Long: +DI > -DI and price above EMA20 and price above 1-day EMA50
            if adx[i] > 25 and plus_di[i] > minus_di[i] and close[i] > ema_20[i] and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: -DI > +DI and price below EMA20 and price below 1-day EMA50
            elif adx[i] > 25 and minus_di[i] > plus_di[i] and close[i] < ema_20[i] and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals