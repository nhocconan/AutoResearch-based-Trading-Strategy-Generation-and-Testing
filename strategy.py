#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams Fractals: Bearish fractal (high) for short signals, bullish fractal (low) for long signals.
- Entry: Long when price breaks above bullish fractal AND 1d EMA50 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below bearish fractal AND 1d EMA50 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below bearish fractal,
        exit short when price crosses above bullish fractal.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Williams Fractals provide structural breakout levels that work in both trending and ranging markets.
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
    
    # Get 1d data for EMA50 trend filter and Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Get volume MA(20) for confirmation (using 12h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    # Williams Fractals need 2 extra bars for confirmation (completed fractal)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above bullish fractal AND 1d EMA50 bullish AND volume confirmed
            if curr_close > bullish_fractal_aligned[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bearish fractal AND 1d EMA50 bearish AND volume confirmed
            elif curr_close < bearish_fractal_aligned[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below bearish fractal (reversion to mean)
            if curr_close < bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above bullish fractal (reversion to mean)
            if curr_close > bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0