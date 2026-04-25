#!/usr/bin/env python3
"""
6h_WilliamsFractal_1dTrend_VolumeBreakout
Hypothesis: Williams Fractals on 1d identify key swing highs/lows. A break above the most recent bullish fractal (with volume spike and 1d uptrend) signals continuation long; break below bearish fractal (with volume spike and 1d downtrend) signals continuation short. Uses discrete position sizing (0.25) to limit fee drag and drawdown. Works in trending markets by capturing breakouts from swing points. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Williams fractals and EMA34 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Williams fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values, df_1d['low'].values
    )
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (6f) with extra delay for fractals (need 2 extra 1d bars for confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start index: need volume MA (20) + aligned HTF arrays
    start_idx = max(20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal with volume spike and 1d uptrend
            long_breakout = (curr_close > bullish_fractal_aligned[i]) and vol_spike[i] and (curr_close > ema_34_1d_aligned[i])
            # Short: price breaks below bearish fractal with volume spike and 1d downtrend
            short_breakout = (curr_close < bearish_fractal_aligned[i]) and vol_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below bearish fractal OR trend turns down
            if (curr_close < bearish_fractal_aligned[i]) or (curr_close < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above bullish fractal OR trend turns up
            if (curr_close > bullish_fractal_aligned[i]) or (curr_close > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0