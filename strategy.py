#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
Long when price breaks above most recent bullish fractal AND 1d close > 1d EMA34 AND 6h volume > 1.5x 20-period average volume.
Short when price breaks below most recent bearish fractal AND 1d close < 1d EMA34 AND 6h volume > 1.5x 20-period average volume.
Exit when price reaches midpoint of fractal levels OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~15-30 trades/year on 6h timeframe.
Combines price structure (Williams Fractals), trend filter (1d EMA34), and volume confirmation for robustness.
Fractals require 2-bar confirmation after the center bar to avoid look-ahead.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals on 1d data (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Align fractals with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 6h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    bullish_fractal_level = 0.0  # most recent bullish fractal level
    bearish_fractal_level = 0.0  # most recent bearish fractal level
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 and volMA need 34 and 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_34_1d_aligned[i]
        
        # Update most recent fractal levels (only when valid)
        if not np.isnan(bullish_fractal_aligned[i]):
            bullish_fractal_level = bullish_fractal_aligned[i]
        if not np.isnan(bearish_fractal_aligned[i]):
            bearish_fractal_level = bearish_fractal_aligned[i]
        
        if position == 0:
            # Long: Break above bullish fractal AND bullish trend (1d close > EMA34) AND volume spike
            if (bullish_fractal_level > 0 and 
                close[i] > bullish_fractal_level and 
                ema_34_1d_aligned[i] > ema_val and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below bearish fractal AND bearish trend (1d close < EMA34) AND volume spike
            elif (bearish_fractal_level > 0 and 
                  close[i] < bearish_fractal_level and 
                  ema_34_1d_aligned[i] < ema_val and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Calculate midpoint of fractal levels for exit
            fractal_mid = 0.0
            if bullish_fractal_level > 0 and bearish_fractal_level > 0:
                fractal_mid = (bullish_fractal_level + bearish_fractal_level) / 2
            elif bullish_fractal_level > 0:
                fractal_mid = bullish_fractal_level / 2
            elif bearish_fractal_level > 0:
                fractal_mid = bearish_fractal_level / 2
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price reaches fractal midpoint
            if position == 1 and fractal_mid > 0 and close[i] >= fractal_mid:
                exit_signal = True
            elif position == -1 and fractal_mid > 0 and close[i] <= fractal_mid:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bullish_fractal_level = 0.0
                bearish_fractal_level = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeConfirmation_FractalMid_EXIT_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0