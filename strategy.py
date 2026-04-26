#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_Volume_v1
Hypothesis: Use weekly Williams Fractals as dynamic support/resistance levels.
Break above weekly bearish fractal with volume confirmation and 1w uptrend = long.
Break below weekly bullish fractal with volume confirmation and 1w downtrend = short.
1d timeframe minimizes trade frequency (<25/year) while capturing major breaks.
Works in bull (breakouts continue) and bear (breakdowns continue) via trend filter.
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
    
    # Get 1w data for fractals and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1w (requires 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Williams fractals need 2 extra 1w bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of fractal lookback (5), EMA(34), volume MA(20)
    start_idx = max(5, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        regime_long = close_val > ema_34_1w_aligned[i]  # 1w uptrend
        regime_short = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: Close breaks above weekly bearish fractal AND volume confirm AND 1w uptrend
            long_signal = (close_val > bearish_fractal_aligned[i]) and vol_conf and regime_long
            
            # Short: Close breaks below weekly bullish fractal AND volume confirm AND 1w downtrend
            short_signal = (close_val < bullish_fractal_aligned[i]) and vol_conf and regime_short
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Close crosses below weekly bullish fractal OR 1w trend flips down
            if (close_val < bullish_fractal_aligned[i]) or (not regime_long):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Close crosses above weekly bearish fractal OR 1w trend flips up
            if (close_val > bearish_fractal_aligned[i]) or (not regime_short):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0