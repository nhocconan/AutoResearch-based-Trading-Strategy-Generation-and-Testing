#!/usr/bin/env python3
"""
12h Williams Fractal Breakout with Daily EMA Trend and Volume Spike
Hypothesis: Williams fractals identify significant swing highs/lows that act as support/resistance.
Breakouts above bearish fractals or below bullish fractals with daily EMA trend alignment and
volume spikes capture strong moves in both bull and bear markets. Uses 12h timeframe with 1d HTF
for trend and fractal confirmation. Targets 50-150 trades over 4 years (12-37/year) to avoid fee drag.
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
    
    # Get 1d data for fractals and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d (requires 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Williams fractals need 2 extra bars for confirmation (right side)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 34-period EMA on 1d close (only needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 20-period volume MA for 12h volume spike
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for volume MA and fractals (2 extra delay already in alignment)
    start_idx = max(20, 4)  # 20 for volume MA, 4 for fractal calculation (2 left + center + 2 right)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vol_ma_20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period 12h average
        volume_confirm = curr_volume > 1.5 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above bearish fractal (resistance), above 1d EMA, volume confirmation
            long_entry = (curr_close > bear_fract and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below bullish fractal (support), below 1d EMA, volume confirmation
            short_entry = (curr_close < bull_fract and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below bullish fractal (support) OR below 1d EMA
            if curr_close < bull_fract or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above bearish fractal (resistance) OR above 1d EMA
            if curr_close > bear_fract or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_DailyEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0