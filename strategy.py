#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1w EMA34 trend filter and volume confirmation
# Uses discrete sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years (12-37/year).
# Williams Fractal identifies potential reversal points; breakout above/below confirms direction.
# 1w EMA34 filters counter-trend moves on weekly timeframe.
# Volume spike ensures institutional participation. Works in both bull and bear via trend filter.

name = "12h_WilliamsFractal_1wEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractal: Bearish = high[i] is highest of 5 bars (i-2 to i+2)
    # Bullish = low[i] is lowest of 5 bars (i-2 to i+2)
    # We need 2-bar confirmation after the center bar
    n_fractal = 5
    half = n_fractal // 2  # 2
    
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(half, n - half):
        # Check if high[i] is the highest in window [i-2, i+2]
        window_high = high[i-half:i+half+1]
        if high[i] == np.max(window_high):
            bearish_fractal[i] = True
        # Check if low[i] is the lowest in window [i-2, i+2]
        window_low = low[i-half:i+half+1]
        if low[i] == np.min(window_low):
            bullish_fractal[i] = True
    
    # Calculate 1w EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Williams fractal needs 2 extra 1w bars after center bar for confirmation
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced for trade frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, half*2+2, 34, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with fractal break and 1w EMA34 trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above bullish fractal level + close above 1w EMA34
                if bullish_fractal[i] and curr_close > curr_low and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below bearish fractal level + close below 1w EMA34
                elif bearish_fractal[i] and curr_close < curr_high and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below entry low OR loses 1w trend
            if curr_low <= stop_loss or curr_close < curr_low or curr_close < curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above entry high OR loses 1w trend
            if curr_high >= stop_loss or curr_close > curr_high or curr_close > curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals