#!/usr/bin/env python3
"""
6h_1d_Williams_Fractal_Momentum
Hypothesis: Williams fractals on 1d identify turning points. Enter long on bullish fractal breakout above prior high with 6h momentum confirmation; short on bearish fractal break below prior low. Works in bull (breakouts) and bear (breakdowns) via directional fractal breaks. Volume filter avoids false breaks. Target: 15-25 trades/year.
"""

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
    
    # Get daily data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams fractals on daily
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Need 2 extra bars for confirmation after the center bar
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 6h momentum: 12-period RSI > 50 for long, < 50 for short
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_ma = pd.Series(rsi_values).rolling(window=12, min_periods=12).mean().values
    
    # Volume filter: current volume > 1.5x 24-period average (4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    volume_expansion = volume > (vol_ma_24 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or
            np.isnan(rsi_ma[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: bullish fractal confirmed and price breaks above prior day high with momentum
        long_condition = (bullish_fractal_confirmed[i] > 0) and (high[i] > high_1d[-1] if len(high_1d) > 0 else False) and (rsi_ma[i] > 50) and volume_expansion[i]
        # Short: bearish fractal confirmed and price breaks below prior day low with momentum
        short_condition = (bearish_fractal_confirmed[i] > 0) and (low[i] < low_1d[-1] if len(low_1d) > 0 else False) and (rsi_ma[i] < 50) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        else:
            # Exit: momentum divergence or loss of volume expansion
            if position == 1 and (rsi_ma[i] < 50 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (rsi_ma[i] > 50 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Williams_Fractal_Momentum"
timeframe = "6h"
leverage = 1.0