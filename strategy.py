#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v3
# Hypothesis: Use Williams Fractal breakouts on 4h in direction of 1d EMA trend, filtered by volume spikes.
# Williams Fractals identify potential reversal/continuation points; breakouts above/below recent fractals
# with volume confirmation and 1d trend alignment capture momentum moves. Works in bull/bear by following
# higher timeframe trend. Target: 20-50 trades/year to minimize fee drag.

name = "4h_fractal_breakout_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 4h
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Need 2 extra bars for confirmation (fractal forms at n-2, needs 2 more closes)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # Daily EMA trend filter (34-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 2.0x 30-period average (~5 days)
    vol_period = 30
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(5, vol_period) + 5  # fractal needs 5 bars, plus volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below bullish fractal or trend fails
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above bearish fractal or trend fails
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Breakout long: price breaks above bullish fractal with uptrend
                if close[i] > bullish_fractal_aligned[i] and close[i] > ema_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below bearish fractal with downtrend
                elif close[i] < bearish_fractal_aligned[i] and close[i] < ema_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals