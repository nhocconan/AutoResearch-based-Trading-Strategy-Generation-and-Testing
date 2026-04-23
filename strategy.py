#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1-week Williams Fractals + 1d EMA34 trend filter + volume confirmation.
Long when bullish fractal forms AND price > 1d EMA34 AND volume > 1.5x 20-period average.
Short when bearish fractal forms AND price < 1d EMA34 AND volume > 1.5x 20-period average.
Exit when price retouches 1d EMA34 or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Williams Fractals provide leading reversal signals from higher timeframe, EMA34 filters trend direction,
volume confirmation reduces false breakouts. Works in both bull and bear markets by requiring
confluence of fractal, trend, and volume.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1-week Williams Fractals (need 2 extra bars for confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Add 2-bar delay for fractal confirmation (needs 2 future 1w bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_34 = ema_34_1d_aligned[i]
        bearish = bearish_fractal_aligned[i]
        bullish = bullish_fractal_aligned[i]
        
        if position == 0:
            # Long: Bullish fractal AND price > 1d EMA34 AND volume spike
            if (bullish and price > ema_34 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Bearish fractal AND price < 1d EMA34 AND volume spike
            elif (bearish and price < ema_34 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches 1d EMA34
            if position == 1 and price <= ema_34:
                exit_signal = True
            elif position == -1 and price >= ema_34:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractals_1dEMA34_VolumeConfirmation_ATRStop"
timeframe = "6h"
leverage = 1.0