#!/usr/bin/env python3
# Hypothesis: 12h Williams Fractal breakout with 1w trend filter and volume confirmation.
# Bullish breakout: price breaks above recent Williams bearish fractal (resistance) AND 1w EMA50 uptrend AND volume > 1.5x average
# Bearish breakout: price breaks below recent Williams bullish fractal (support) AND 1w EMA50 downtrend AND volume > 1.5x average
# Exit when price retraces to midpoint of last fractal level OR trend reverses
# Uses 12h timeframe for lower frequency, Williams Fractals for key support/resistance, 1w EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via breakdown continuation.

name = "12h_WilliamsFractal_1wTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Fractals and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Williams Fractals on 12h
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_12h, low_12h)
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    last_bearish_fractal = np.nan  # last confirmed bearish fractal (resistance)
    last_bullish_fractal = np.nan  # last confirmed bullish fractal (support)
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Update last confirmed fractal levels
        if not np.isnan(bearish_fractal_aligned[i]):
            last_bearish_fractal = bearish_fractal_aligned[i]
        if not np.isnan(bullish_fractal_aligned[i]):
            last_bullish_fractal = bullish_fractal_aligned[i]
        
        if position == 0:
            # LONG: price breaks above recent bearish fractal (resistance) AND 1w EMA50 uptrend AND volume confirmation
            if (not np.isnan(last_bearish_fractal) and 
                close[i] > last_bearish_fractal and 
                close[i] > ema50_1w_aligned[i] and 
                volume_filter_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below recent bullish fractal (support) AND 1w EMA50 downtrend AND volume confirmation
            elif (not np.isnan(last_bullish_fractal) and 
                  close[i] < last_bullish_fractal and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_filter_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to midpoint of last fractal level OR trend reversal (price < 1w EMA50)
            if (not np.isnan(last_bullish_fractal) and not np.isnan(last_bearish_fractal) and
                close[i] <= (last_bullish_fractal + last_bearish_fractal) / 2) or \
               close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to midpoint of last fractal level OR trend reversal (price > 1w EMA50)
            if (not np.isnan(last_bullish_fractal) and not np.isnan(last_bearish_fractal) and
                close[i] >= (last_bullish_fractal + last_bearish_fractal) / 2) or \
               close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals