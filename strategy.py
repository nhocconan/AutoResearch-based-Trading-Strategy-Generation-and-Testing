#!/usr/bin/env python3
"""
1d Williams Fractal Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Williams Fractals identify key swing points on daily chart. Breakouts above
bearish fractals (or below bullish fractals) with weekly EMA50 trend alignment, volume
confirmation, and choppiness regime filter capture strong momentum moves while avoiding
choppy markets. Works in bull/bear via higher timeframe trend filter and regime avoidance.
Target: 10-25 trades/year on 1d to minimize fee drag.
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
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load daily data for Williams Fractals and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams Fractals on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Align with 2-bar delay for fractal confirmation (needs 2 future daily bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Daily choppiness index (14-period)
    chop_period = 14
    atr_1d = pd.Series(np.maximum.reduce([
        high_1d[1:] - low_1d[1:],
        np.abs(high_1d[1:] - close_1d[:-1]),
        np.abs(low_1d[1:] - close_1d[:-1])
    ]), index=df_1d.index[1:]).rolling(window=chop_period, min_periods=chop_period).mean().values
    # Pad ATR array to match length
    atr_1d_padded = np.full(len(df_1d), np.nan)
    atr_1d_padded[1:] = atr_1d
    sum_high_low = pd.Series(high_1d - low_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    chop = 100 * np.log10(sum_high_low / (chop_period * atr_1d_padded)) / np.log10(chop_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Daily volume confirmation: current volume > 2.0 * 20-period average
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (vol_ma_1d_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20, 14) + 2  # EMA50 + volMA20 + chop14 + fractal delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        # Choppiness filter: avoid ranging markets (CHOP > 61.8) and extreme trends (CHOP < 38.2)
        chop_value = chop_aligned[i]
        chop_filter = (chop_value >= 38.2) & (chop_value <= 61.8)
        
        # Fractal breakout conditions
        # Long: price breaks above bearish fractal (resistance)
        long_breakout = curr_high > bearish_fractal_aligned[i]
        # Short: price breaks below bullish fractal (support)
        short_breakout = curr_low < bullish_fractal_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: breakout + trend + volume + chop filter
            long_entry = long_breakout and bullish_bias and vol_spike and chop_filter
            short_entry = short_breakout and bearish_bias and vol_spike and chop_filter
            
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
            # Exit: price breaks below bullish fractal (support) OR loss of bullish bias
            if (curr_low < bullish_fractal_aligned[i]) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above bearish fractal (resistance) OR loss of bearish bias
            if (curr_high > bearish_fractal_aligned[i]) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0