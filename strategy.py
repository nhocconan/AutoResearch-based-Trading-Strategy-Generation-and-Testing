#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 12h EMA34 Trend + Volume Spike
Hypothesis: Williams fractals identify key swing points. Breakout above latest bullish fractal or below bearish fractal with volume confirmation and 12h EMA34 trend filter captures momentum moves. Works in bull (long on upside fractal break) and bear (short on downside fractal break). Volume spike ensures participation. Target: 15-30 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Williams fractals (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 12h
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values,
    )
    # Fractals need 2 extra 12h bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > bullish fractal, above 12h EMA34, volume confirmation
            long_entry = (curr_close > bull_fractal) and (curr_close > ema_34_val) and volume_confirm
            # Short: price < bearish fractal, below 12h EMA34, volume confirmation
            short_entry = (curr_close < bear_fractal) and (curr_close < ema_34_val) and volume_confirm
            
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
            # Exit: price crosses below 12h EMA34 OR bearish fractal break (stop and reverse)
            if curr_close < ema_34_val or curr_close < bear_fractal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above 12h EMA34 OR bullish fractal break (stop and reverse)
            if curr_close > ema_34_val or curr_close > bull_fractal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0