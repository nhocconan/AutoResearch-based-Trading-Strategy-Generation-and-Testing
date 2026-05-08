#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w volume spike and 1w EMA50 trend filter.
# Long when price breaks above recent bullish fractal AND 1w volume > 2.0x 24-period average AND price > 1w EMA50.
# Short when price breaks below recent bearish fractal AND 1w volume > 2.0x 24-period average AND price < 1w EMA50.
# Exit when price crosses back below/above 1w EMA50 (trend-based exit).
# Williams Fractals require 2-bar confirmation, so we use additional_delay_bars=2 in align_htf_to_ltf.
# Target: 30-100 total trades over 4 years (7-25/year) for low fee drift.

name = "1d_WilliamsFractal_Breakout_1wVolume_1wEMA50"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter and volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1w data
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Williams fractals need 2 extra 1w bars after the center bar for confirmation
    bearish_fractal_1d = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_1d = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # 1w volume filter: current volume > 2.0x 24-period average
    vol_ma24 = pd.Series(df_1w['volume'].values).rolling(window=24, min_periods=24).mean().values
    volume_filter_1d = align_htf_to_ltf(prices, df_1w, vol_ma24 > 2.0)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for EMA and fractals
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_1d[i]) or np.isnan(bullish_fractal_1d[i]) or 
            np.isnan(volume_filter_1d[i]) or np.isnan(ema50_1w_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above bullish fractal, volume spike, above 1w EMA50
            long_cond = (close[i] > bullish_fractal_1d[i]) and volume_filter_1d[i] and (close[i] > ema50_1w_1d[i])
            # Short conditions: price breaks below bearish fractal, volume spike, below 1w EMA50
            short_cond = (close[i] < bearish_fractal_1d[i]) and volume_filter_1d[i] and (close[i] < ema50_1w_1d[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1w EMA50 (trend change)
            if close[i] < ema50_1w_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1w EMA50 (trend change)
            if close[i] > ema50_1w_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals