#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
Long when bullish fractal breaks above resistance AND 1w EMA50 rising AND 1d volume > 2.0x 20-period MA.
Short when bearish fractal breaks below support AND 1w EMA50 falling AND 1d volume > 2.0x 20-period MA.
Exit when price touches opposite fractal level or 1w EMA50 reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades in bear markets, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams Fractals provide reliable swing points, 1w EMA50 filters major trend, volume spike avoids low-momentum breakouts.
Works in bull (trend filters) and bear (volume spikes on breakdowns from swing highs/lows).
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
    
    # Calculate Williams Fractals on 6h data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Fractals need 2 extra 6h bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume MA (20-period) for spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50, volume MA (fractals already aligned)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        # Get current fractal values (already aligned and delayed)
        bullish_val = bullish_fractal_aligned[i]
        bearish_val = bearish_fractal_aligned[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1d volume > 2.0x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Bullish fractal resistance break AND EMA50 rising AND volume filter
            if not np.isnan(bullish_val) and price > bullish_val and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal support break AND EMA50 falling AND volume filter
            elif not np.isnan(bearish_val) and price < bearish_val and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches bearish fractal support OR EMA50 starts falling
                if not np.isnan(bearish_val) and price < bearish_val:
                    exit_signal = True
                elif i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches bullish fractal resistance OR EMA50 starts rising
                if not np.isnan(bullish_val) and price > bullish_val:
                    exit_signal = True
                elif i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0