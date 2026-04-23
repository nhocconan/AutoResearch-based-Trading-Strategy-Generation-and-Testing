#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation.
- Bollinger Bands(20,2): Squeeze when BB width < 20th percentile of last 50 periods
- Breakout long: close breaks above upper BB + volume > 2x 20-period avg + price > 1w EMA50
- Breakout short: close breaks below lower BB + volume > 2x 20-period avg + price < 1w EMA50
- Exit: Opposite BB break or BB width > 80th percentile (squeeze end)
- Uses Bollinger squeeze for low volatility breakouts, volume for conviction, 1w EMA50 for HTF trend
- Works in bull (breakouts with trend) and bear (breakdowns with trend) markets
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
"""

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
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    bb_width = (upper_bb - lower_bb) / sma  # Normalized width
    
    # BB width percentiles for squeeze detection (50 lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_20th = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    bb_width_80th = bb_width_series.rolling(window=50, min_periods=20).quantile(0.80).values
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 50, 20)  # Need 50 for BB width percentiles, 20 for volume MA, 20 for BB
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or 
            np.isnan(bb_width_20th[i]) or np.isnan(bb_width_80th[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB width < 20th percentile (low volatility)
        is_squeeze = bb_width[i] < bb_width_20th[i]
        
        # Volume confirmation (> 2x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > upper_bb[i-1]  # Break above upper BB (using previous bar)
        short_breakout = close[i] < lower_bb[i-1]  # Break below lower BB
        
        # Exit conditions
        exit_long = (close[i] < lower_bb[i]) or (bb_width[i] > bb_width_80th[i])  # Break below lower BB or squeeze end
        exit_short = (close[i] > upper_bb[i]) or (bb_width[i] > bb_width_80th[i])  # Break above upper BB or squeeze end
        
        if position == 0:
            # Look for breakout from squeeze
            if is_squeeze and volume_confirm:
                if long_breakout and close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout and close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_Breakout_1wEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0