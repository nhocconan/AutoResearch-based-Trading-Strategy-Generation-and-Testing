#!/usr/bin/env python3
"""
1d Williams Fractal Breakout with Weekly EMA34 Trend and Volume Spike
Hypothesis: Williams fractals identify key support/resistance levels. A breakout above a
bearish fractal (or below a bullish fractal) with weekly uptrend/downtrend and volume spike
signals trend continuation. Uses 1d timeframe with 1w HTF for trend. Targets 30-100 total trades
over 4 years (7-25/year). Works in both bull and bear markets by following the weekly trend.
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
    
    # Get weekly data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on weekly close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    # Weekly EMA needs no extra delay as it's trend-following on completed weekly candles
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Compute Williams fractals on daily data
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Fractals need 2 extra weekly bars for confirmation after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 20-period volume MA for daily volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for fractals and volume MA
    start_idx = max(20, 2)  # 20 for volume MA, 2 for fractals (need 2 bars ahead)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current daily volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Breakout conditions
        # Bullish breakout: price breaks above bearish fractal resistance
        bullish_breakout = curr_close > bearish_fractal_val
        # Bearish breakout: price breaks below bullish fractal support
        bearish_breakout = curr_close < bullish_fractal_val
        
        if position == 0:
            # Look for entry signals
            # Long: Bullish breakout AND weekly uptrend (price > EMA34) AND volume confirmation
            long_entry = bullish_breakout and (curr_close > ema_trend) and volume_confirm
            # Short: Bearish breakout AND weekly downtrend (price < EMA34) AND volume confirmation
            short_entry = bearish_breakout and (curr_close < ema_trend) and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below bullish fractal support OR weekly trend turns down
            if curr_close < bullish_fractal_val or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above bearish fractal resistance OR weekly trend turns up
            if curr_close > bearish_fractal_val or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Fractal_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0