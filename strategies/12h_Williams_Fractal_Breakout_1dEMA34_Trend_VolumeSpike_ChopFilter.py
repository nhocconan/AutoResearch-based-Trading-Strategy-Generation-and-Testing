#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Williams Fractals on 1d identify key swing points; breakouts above/below these levels with 1d EMA34 trend filter, volume confirmation, and choppiness regime filter capture strong momentum moves while avoiding whipsaws in ranging markets. Designed for 12h timeframe to target 12-37 trades/year, minimizing fee drag. Works in both bull and bear markets by following the daily trend and using chop filter to avoid false signals in low-volatility regimes.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Fractals on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Choppiness Index filter (14-period) - avoid ranging markets
    # CHOP > 61.8 = ranging/choppy, CHOP < 38.2 = trending
    # We only trade when CHOP < 61.8 (not strongly choppy)
    hl_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum()
    true_range = pd.Series(np.maximum(high - low, 
                                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                                np.abs(np.roll(close, 1) - low)))).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(hl_range / true_range) / np.log10(14)
    chop_values = chop.values
    not_choppy = chop_values < 61.8  # Allow trading when not excessively choppy
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(30, 34, 14)  # volume MA, EMA, chop
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_not_choppy = not_choppy[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above bullish fractal AND bullish bias AND volume spike AND not choppy
            long_entry = (curr_high > bullish_fractal_aligned[i]) and bullish_bias and vol_spike and is_not_choppy
            # Short: price breaks below bearish fractal AND bearish bias AND volume spike AND not choppy
            short_entry = (curr_low < bearish_fractal_aligned[i]) and bearish_bias and vol_spike and is_not_choppy
            
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
            # Exit: price falls below bearish fractal (mean reversion) OR loss of bullish bias OR too choppy
            if (curr_low < bearish_fractal_aligned[i]) or (curr_close < ema_1d_aligned[i]) or (not is_not_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above bullish fractal (mean reversion) OR loss of bearish bias OR too choppy
            if (curr_high > bullish_fractal_aligned[i]) or (curr_close > ema_1d_aligned[i]) or (not is_not_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0