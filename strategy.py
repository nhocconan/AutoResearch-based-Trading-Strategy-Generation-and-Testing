#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Williams Fractals identify significant swing points where price reverses
# Breakouts above/below recent fractal levels capture strong momentum moves
# 1d EMA34 ensures trades only with intermediate-term trend, reducing false breakouts
# Volume confirmation at 2.0x average filters low-participation moves
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 to balance profit potential and fee drag
# Works in both bull (breakouts continue trend) and bear (breakouts capture retracements)

name = "12h_WilliamsFractal_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals: 5-bar pattern (requires 2 bars on each side)
    # Bearish fractal: high[n-2] is highest of [n-4, n-3, n-2, n-1, n]
    # Bullish fractal: low[n-2] is lowest of [n-4, n-3, n-2, n-1, n]
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Calculate bearish fractals (swing highs)
    bearish_fractal = (
        (high_series.shift(2) == high_series.rolling(window=5, center=False).max()) &
        (high_series.shift(2) > high_series.shift(1)) &
        (high_series.shift(2) > high_series.shift(3)) &
        (high_series.shift(2) > high_series.shift(4)) &
        (high_series.shift(2) > high_series)
    ).astype(float) * high_series.shift(2)  # Store the fractal level value
    
    # Calculate bullish fractals (swing lows)
    bullish_fractal = (
        (low_series.shift(2) == low_series.rolling(window=5, center=False).min()) &
        (low_series.shift(2) < low_series.shift(1)) &
        (low_series.shift(2) < low_series.shift(3)) &
        (low_series.shift(2) < low_series.shift(4)) &
        (low_series.shift(2) < low_series)
    ).astype(float) * low_series.shift(2)  # Store the fractal level value
    
    bearish_fractal_values = bearish_fractal.values
    bullish_fractal_values = bullish_fractal.values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams fractals need 2 extra 1d bars for confirmation (formation + 2 bars after)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal_values, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal_values, additional_delay_bars=2
    )
    
    # Volume confirmation: 2.0x 20-period average (stricter threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above bearish fractal (resistance) AND price > 1d EMA34 AND volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bullish fractal (support) AND price < 1d EMA34 AND volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below bullish fractal (support) OR closes below 1d EMA34
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above bearish fractal (resistance) OR closes above 1d EMA34
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals