#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_12hTrend_ATRStop_v1
Hypothesis: Williams fractal breakouts on 6h with 12h EMA trend filter and ATR-based stoploss.
Only trade bullish fractal breaks above R3 (short-term resistance) in uptrend or bearish fractal breaks below S3 (short-term support) in downtrend.
Uses Williams fractal to identify key swing points, aligned with 12h EMA direction for trend filtering.
Targets 12-37 trades/year on 6h timeframe, avoiding fee drag while capturing momentum from fractal breakouts.
Uses discrete sizing (0.25) and ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
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
    
    # Get 6h data for Williams fractals - primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Williams fractals on 6h
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_6h, low_6h)
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_6h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_6h, bullish_fractal, additional_delay_bars=2)
    
    # Get 12h data for EMA trend filter - HTF
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate ATR(14) for stoploss on 6h
    # True Range
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_6h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(100, 34, 21)  # Fractals need ~100, EMA needs 34, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(atr_6h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        bearish_val = bearish_fractal_aligned[i]
        bullish_val = bullish_fractal_aligned[i]
        ema_val = ema_12h_aligned[i]
        atr_val = atr_6h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        if position == 0:
            # Look for entry signals: Williams fractal breakout with trend confirmation
            # Long: price breaks above bullish fractal (resistance), above 12h EMA
            long_signal = (high_val > bullish_val) and (close_val > ema_val)
            # Short: price breaks below bearish fractal (support), below 12h EMA
            short_signal = (low_val < bearish_val) and (close_val < ema_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below 12h EMA
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above 12h EMA
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hTrend_ATRStop_v1"
timeframe = "6h"
leverage = 1.0