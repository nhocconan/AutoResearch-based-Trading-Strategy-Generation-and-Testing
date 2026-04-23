#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above recent bullish fractal AND 1d EMA34 rising AND volume > 2.0x 20-period MA.
Short when price breaks below recent bearish fractal AND 1d EMA34 falling AND volume > 2.0x 20-period MA.
Exit when price touches opposite fractal level or 1d EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Williams Fractals provide swing high/low structure, 1d EMA34 filters major trend.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Calculate Williams Fractals on 1d timeframe (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Compute Williams Fractals (requires 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA34 for trend filter (HTF)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34, volume MA (fractals handled by alignment)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Get most recent fractal levels (forward-filled)
        recent_bullish = bullish_fractal_aligned[i]
        recent_bearish = bearish_fractal_aligned[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Price above recent bullish fractal AND EMA34 rising AND volume filter
            if not np.isnan(recent_bullish) and price > recent_bullish and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price below recent bearish fractal AND EMA34 falling AND volume filter
            elif not np.isnan(recent_bearish) and price < recent_bearish and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches/below recent bearish fractal OR EMA34 starts falling
                if (not np.isnan(recent_bearish) and price < recent_bearish) or \
                   (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches/above recent bullish fractal OR EMA34 starts rising
                if (not np.isnan(recent_bullish) and price > recent_bullish) or \
                   (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0