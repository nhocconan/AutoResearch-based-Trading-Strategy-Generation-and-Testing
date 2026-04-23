#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above recent bullish fractal AND 1w EMA50 rising AND 1d volume > 1.5x 20-period MA.
Short when price breaks below recent bearish fractal AND 1w EMA50 falling AND 1d volume > 1.5x 20-period MA.
Exit when price touches opposite fractal level or 1w EMA50 reverses.
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Williams Fractals identify key swing points, 1w EMA50 filters major trend, volume avoids low-momentum breakouts.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate Williams Fractals on 1d data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Williams fractals need 2 extra 1d bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA, EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_1w_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1d volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above bullish fractal AND EMA50 rising AND volume filter
            if not np.isnan(bullish_fractal_val) and price > bullish_fractal_val and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below bearish fractal AND EMA50 falling AND volume filter
            elif not np.isnan(bearish_fractal_val) and price < bearish_fractal_val and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches bearish fractal OR EMA50 starts falling
                if not np.isnan(bearish_fractal_val) and price < bearish_fractal_val:
                    exit_signal = True
                elif i >= start_idx + 1 and ema_val < ema_50_1w_aligned[i-1]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches bullish fractal OR EMA50 starts rising
                if not np.isnan(bullish_fractal_val) and price > bullish_fractal_val:
                    exit_signal = True
                elif i >= start_idx + 1 and ema_val > ema_50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0