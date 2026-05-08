#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Donchian_20_WeeklyTrend_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: price above/below 20-week EMA
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend = (close_1w > ema20_1w).astype(float)  # 1 = uptrend, 0 = downtrend
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Weekly volume confirmation: current weekly volume > 1.5 * 10-week average
    volume_1w = df_1w['volume'].values
    vol_ma10w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_conf_1w = volume_1w > (vol_ma10w * 1.5)
    vol_conf_aligned = align_htf_to_ltf(prices, df_1w, vol_conf_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Donchian and weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_conf_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 20-day high with weekly uptrend and volume confirmation
            long_cond = (close[i] > high_20[i] and weekly_trend_aligned[i] > 0.5 and vol_conf_aligned[i])
            
            # Short entry: price breaks below 20-day low with weekly downtrend and volume confirmation
            short_cond = (close[i] < low_20[i] and weekly_trend_aligned[i] < 0.5 and vol_conf_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below 20-day low (mean reversion)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above 20-day high (mean reversion)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on daily timeframe with weekly trend and volume filters.
# Works in bull markets (trend-following breakouts) and bear markets (mean reversion at opposite band).
# Weekly EMA20 ensures alignment with longer-term trend, reducing counter-trend trades.
# Weekly volume confirmation (1.5x 10-week average) ensures institutional participation.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.