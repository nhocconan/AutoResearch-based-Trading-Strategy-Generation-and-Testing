#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1wTrend_Filter_v1
Hypothesis: On 6h timeframe, Williams Fractal breakouts (bullish/bearish) with 1-week EMA34 trend filter and volume confirmation (>1.3x avg) provides high-probability directional signals. Williams Fractals identify key swing points where price respects structure. Long when price breaks above latest bearish fractal (resistance) + price > 1w EMA34 + volume spike; short when price breaks below latest bullish fractal (support) + price < 1w EMA34 + volume spike. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 total trades over 4 years (12-37/year) for optimal 6h frequency. Works in bull markets (trend following) and bear markets (trend following with short signals). Volume confirmation ensures institutional participation, reducing false signals in low-volume environments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF indicators (EMA34 trend filter, Williams Fractals)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Fractals on 1w data
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
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA + fractal calculation
    start_idx = max(34, 20, 50)  # EMA34 + volume MA + fractal lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_confirmed = vol_ratio[i] > 1.3  # volume at least 1.3x average
        
        if position == 0:
            # Long: price > 1w EMA34 + breaks above bearish fractal (resistance) + volume
            long_signal = (close[i] > ema_34_1w_aligned[i] and 
                          close[i] > bearish_fractal_aligned[i] and 
                          vol_confirmed)
            
            # Short: price < 1w EMA34 + breaks below bullish fractal (support) + volume
            short_signal = (close[i] < ema_34_1w_aligned[i] and 
                           close[i] < bullish_fractal_aligned[i] and 
                           vol_confirmed)
            
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
            # Exit: price closes below 1w EMA34 OR breaks below bullish fractal (support)
            if close[i] < ema_34_1w_aligned[i] or close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above 1w EMA34 OR breaks above bearish fractal (resistance)
            if close[i] > ema_34_1w_aligned[i] or close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0