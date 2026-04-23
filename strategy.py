#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when price breaks above recent Williams Fractal high AND price > 1d EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below recent Williams Fractal low AND price < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the opposite Williams Fractal level (mean reversion)
- Uses 1d EMA34 for HTF trend alignment to avoid counter-trend trades
- Volume spike ensures institutional participation and reduces false breakouts
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
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
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams Fractals on primary 12h timeframe
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Williams Fractals need 2 extra 12h bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 22, 21)  # Need 34 for EMA34 (34+1), 2 for fractal confirmation (5+2), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using Williams Fractals)
        breakout_up = close[i] > bullish_fractal_aligned[i]  # Break above recent fractal high
        breakout_down = close[i] < bearish_fractal_aligned[i]  # Break below recent fractal low
        
        # Trend filter (using 1d EMA34)
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses the opposite Williams Fractal level (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below bearish fractal (recent low)
                if close[i] < bearish_fractal_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above bullish fractal (recent high)
                if close[i] > bullish_fractal_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0