#!/usr/bin/env python3
"""
4h_Williams_Fractal_Breakout_1d_Trend_v1
Williams Fractal breakout with 1-day EMA trend filter.
- Bearish fractal breakout (price breaks below recent low) triggers short in downtrend.
- Bullish fractal breakout (price breaks above recent high) triggers long in uptrend.
- 1-day EMA34 determines trend: price > EMA34 = uptrend, price < EMA34 = downtrend.
- Uses volume confirmation: require volume > 1.5x 20-period average.
- Fixed stoploss via signal=0 when price reverses past fractal level.
Designed to catch momentum after fractal confirmation in trending markets.
Target: 20-50 trades per year (~80-200 total over 4 years).
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
    
    # === 1-day EMA34 for trend filter (load ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Williams Fractals on 1-day (need 2-bar confirmation) ===
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Additional 2-bar delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # === Volume confirmation (20-period average) ===
    vol_ma = np.zeros_like(volume)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    
    # Track position and entry levels
    position = 0  # 0: flat, 1: long, -1: short
    entry_level = 0.0  # fractal level for stop/re-entry
    
    # Warmup: need enough data for indicators
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(volume[i]) or
            np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        vol_ok = volume[i] >= volume_threshold[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish fractal breakout + uptrend + volume
            if (bullish_fractal_aligned[i] and 
                close[i] > bullish_fractal_aligned[i] and  # price above fractal high
                close[i] > ema34_1d_aligned[i] and         # uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
                entry_level = bullish_fractal_aligned[i]  # use fractal level for stop
                continue
            # Short: bearish fractal breakout + downtrend + volume
            elif (bearish_fractal_aligned[i] and 
                  close[i] < bearish_fractal_aligned[i] and  # price below fractal low
                  close[i] < ema34_1d_aligned[i] and         # downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
                entry_level = bearish_fractal_aligned[i]   # use fractal level for stop
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price closes below entry level (fractal low) OR trend change
            if (close[i] < entry_level or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above entry level (fractal high) OR trend change
            if (close[i] > entry_level or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1d_Trend_v1"
timeframe = "4h"
leverage = 1.0