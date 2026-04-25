#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Williams fractals identify swing points where price reverses; breakouts from recent fractal highs/lows with volume confirmation and EMA trend filter capture momentum moves. Chop filter avoids whipsaws in sideways markets. Works in bull/bear via breakout logic and trend filter.
Target: 12-37 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams fractals on 1d high/low (needs 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 6h Donchian channels for breakout levels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Chop filter: avoid sideways markets
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14.sum() / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_ma = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    chop_filter = chop < chop_ma  # Low chop = trending (good for breakouts)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34+2, 20, 20, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        ch_filter = chop_filter[i]
        ema_trend = ema_34_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = curr_high > highest_high[i-1]  # Break above recent high
        breakout_short = curr_low < lowest_low[i-1]   # Break below recent low
        
        # Fractal conditions: bullish fractal = support, bearish = resistance
        bull_fractal = bullish_fractal_aligned[i-1]  # Recent bullish fractal (support)
        bear_fractal = bearish_fractal_aligned[i-1]  # Recent bearish fractal (resistance)
        
        if position == 0:
            # Look for entry signals - require: breakout + volume + trend + chop filter
            long_entry = breakout_long and vol_spike and (curr_close > ema_trend) and ch_filter
            short_entry = breakout_short and vol_spike and (curr_close < ema_trend) and ch_filter
            
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
            # Exit: price retouches recent low or bearish fractal level
            if curr_low <= lowest_low[i] or curr_close <= bear_fractal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches recent high or bullish fractal level
            if curr_high >= highest_high[i] or curr_close >= bull_fractal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "6h"
leverage = 1.0