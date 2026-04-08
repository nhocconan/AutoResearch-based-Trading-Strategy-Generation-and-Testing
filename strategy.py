#!/usr/bin/env python3
# 6h_williams_fractal_1d_trend_volume_v1
# Hypothesis: Williams Fractal breakout with 1d EMA(50) trend filter and volume confirmation.
# Long when price breaks above recent bearish fractal (resistance) in uptrend with volume.
# Short when price breaks below recent bullish fractal (support) in downtrend with volume.
# Works in bull/bear: trend filter ensures directional alignment, fractal breakouts capture momentum shifts.
# Williams fractals require 2-bar confirmation, so we use additional_delay_bars=2.

name = "6h_williams_fractal_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Align fractals to 6s timeframe with 2-bar confirmation delay
    # Bearish fractal = resistance level (look for upside breakout)
    # Bullish fractal = support level (look for downside breakout)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 1d EMA trend filter (50)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(20, 50) + 5  # volume + EMA + buffer
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below bullish fractal (support) or trend fails
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above bearish fractal (resistance) or trend fails
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above bearish fractal (resistance) with uptrend and volume
            if close[i] > bearish_fractal_aligned[i] and close[i] > ema_1d_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below bullish fractal (support) with downtrend and volume
            elif close[i] < bullish_fractal_aligned[i] and close[i] < ema_1d_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals