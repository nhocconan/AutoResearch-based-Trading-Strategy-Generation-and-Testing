#!/usr/bin/env python3

"""
Hypothesis: Daily Bollinger Band squeeze with weekly trend filter and volume confirmation.
Only trade long when price breaks above upper Bollinger Band during low volatility (squeeze)
and weekly trend is up; short when price breaks below lower Bollinger Band during squeeze
and weekly trend is down. Uses Bollinger Band width percentile to detect squeeze conditions,
avoiding false breakouts in high volatility periods. Designed for low trade frequency
(7-25 trades/year) by requiring multiple confirmations: volatility squeeze, price breakout,
and trend alignment. Works in both bull and bear markets by following the weekly trend.
"""

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
    
    # Bollinger Bands (20, 2) on daily
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (50-period lookback) for squeeze detection
    bb_width_pct = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_width_pct[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(ema34_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: Bollinger Band width in lower 30th percentile
        squeeze = bb_width_pct[i] < 0.3
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: squeeze + price breaks above upper band + weekly uptrend + volume spike
            if squeeze and close[i] > bb_upper[i] and ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: squeeze + price breaks below lower band + weekly downtrend + volume spike
            elif squeeze and close[i] < bb_lower[i] and ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: volatility expansion (end of squeeze) or price returns to middle band
            exit_signal = False
            
            if position == 1:
                # Exit long: volatility expansion or price closes below middle band
                if bb_width_pct[i] > 0.7 or close[i] < bb_middle[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: volatility expansion or price closes above middle band
                if bb_width_pct[i] > 0.7 or close[i] > bb_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Bollinger_Squeeze_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0