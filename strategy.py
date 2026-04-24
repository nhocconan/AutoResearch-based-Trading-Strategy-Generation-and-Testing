#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams Fractals: Bearish fractal (short signal) and bullish fractal (long signal) from 1d data.
- Entry: Long when price breaks above prior 1d bullish fractal AND 1w EMA50 bullish AND volume > 1.5 * volume MA(30).
         Short when price breaks below prior 1d bearish fractal AND 1w EMA50 bearish AND volume > 1.5 * volume MA(30).
- Exit: Close-based reversal - exit long when price crosses below prior 1d bearish fractal,
        exit short when price crosses above prior 1d bullish fractal.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Uses 1w EMA50 trend filter (proven edge from DB top performers) for BTC/ETH/SOL.
- Williams fractals require 2-bar confirmation delay for HTF alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align HTF indicators to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    # Williams fractals need 2 extra bars for confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate volume MA(30) for confirmation (using 12h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 30)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above prior 1d bullish fractal AND 1w EMA50 bullish AND volume confirmed
            if curr_close > bullish_fractal_aligned[i] and curr_close > ema_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 1d bearish fractal AND 1w EMA50 bearish AND volume confirmed
            elif curr_close < bearish_fractal_aligned[i] and curr_close < ema_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 1d bearish fractal (reversion to mean)
            if curr_close < bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 1d bullish fractal (reversion to mean)
            if curr_close > bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0