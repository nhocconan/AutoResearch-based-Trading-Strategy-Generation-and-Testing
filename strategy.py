#!/usr/bin/env python3
"""
1d Williams Fractal Breakout with Weekly EMA34 Trend and Volume Spike
Hypothesis: Williams Fractals identify key swing highs/lows on daily chart.
Breakout above bearish fractal or below bullish fractal with weekly EMA34 trend
alignment and volume confirmation captures momentum in both bull and bear markets.
1d timeframe targets 7-25 trades/year (30-100 total over 4 years).
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
    
    # Load 1d data ONCE before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    
    # Compute Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractals need 2 extra 1d bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Load weekly data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, weekly EMA alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Get fractal levels (already aligned to 1d timeframe)
        bear_level = bearish_fractal_aligned[i]
        bull_level = bullish_fractal_aligned[i]
        
        # Trend filter: price relative to weekly EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: break above bearish fractal (resistance turned support) AND uptrend AND volume spike
            long_entry = (curr_close > bear_level) and uptrend and vol_spike
            # Short: break below bullish fractal (support turned resistance) AND downtrend AND volume spike
            short_entry = (curr_close < bull_level) and downtrend and vol_spike
            
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
            # Exit: price breaks below bullish fractal (support) OR loss of uptrend
            if (curr_close < bull_level) or (curr_close < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above bearish fractal (resistance) OR loss of downtrend
            if (curr_close > bear_level) or (curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0