#!/usr/bin/env python3
"""
1d Williams Fractal Breakout with Weekly EMA34 Trend and Volume Spike
Hypothesis: Williams fractals identify significant swing points. A breakout above a
bearish fractal (or below bullish fractal) with weekly uptrend/downtrend and volume
spike signals trend continuation. Uses 1d timeframe with 1w HTF for trend.
Targets 30-100 total trades over 4 years (7-25/year).
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
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    # Align with extra delay because EMA needs confirmation from completed weekly candle
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Compute Williams fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Align with 2-bar additional delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 20-period volume MA for 1d volume confirmation
    vol_ma_20_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1d[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, fractals, and volume MA
    start_idx = max(34 + 2, 20)  # 34 for EMA + 2 delay, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        vol_ma_1d = vol_ma_20_1d[i]
        
        # Volume confirmation: current 1d volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_1d
        
        if position == 0:
            # Look for entry signals
            # Long: Close above bearish fractal (breakout resistance) AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_close > bear_fractal and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Close below bullish fractal (breakdown support) AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_close < bull_fractal and 
                          curr_close < ema_trend and volume_confirm)
            
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
            # Exit: Close below bullish fractal (breakdown support) OR price falls below EMA34
            if (curr_close < bull_fractal or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Close above bearish fractal (breakout resistance) OR price rises above EMA34
            if (curr_close > bear_fractal or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Fractal_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0