#!/usr/bin/env python3
"""
1d Williams Fractal Breakout + Weekly EMA34 Trend + Volume Spike
Hypothesis: Williams Fractals identify key swing highs/lows where price often reverses or accelerates.
In strong weekly trends (price > weekly EMA34 for longs, price < weekly EMA34 for shorts),
fractal breakouts with volume confirmation capture momentum continuation. Daily timeframe
targets 7-25 trades/year (30-100 over 4 years) to minimize fee drag while capturing major moves.
Works in bull markets (buy fractal high breakouts in uptrend) and bear markets (sell fractal low breakdowns in downtrend).
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter (needs extra delay for confirmation)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w, additional_delay_bars=1)
    
    # Calculate Williams Fractals on daily data (need 2-bar confirmation delay)
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
    
    # Volume confirmation: current volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(50, 34)  # volume MA, weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to weekly EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above bullish fractal AND uptrend AND volume spike
            long_entry = (curr_high > bullish_fractal_aligned[i]) and uptrend and vol_spike
            # Short: price breaks below bearish fractal AND downtrend AND volume spike
            short_entry = (curr_low < bearish_fractal_aligned[i]) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below bearish fractal (potential reversal) OR loss of uptrend
            if (curr_low < bearish_fractal_aligned[i]) or (curr_close < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price breaks above bullish fractal (potential reversal) OR loss of downtrend
            if (curr_high > bullish_fractal_aligned[i]) or (curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_WilliamsFractal_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0