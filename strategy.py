#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WillyFractal_Trend_Range"
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
    
    # Get daily data for fractals, trend, and range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Williams Fractals (need 2 extra bars for confirmation)
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily range: ATR(14) for volatility filter
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align fractals with 2-bar delay (confirmation needed)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Align trend and ATR
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after enough data for EMA50 and ATR
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range filter: only trade when ATR is not too low (avoid chop)
        range_filter = atr14_aligned[i] > 0
        
        if position == 0:
            # Bullish fractal (resistance break) + above trend -> long
            if bullish_fractal_aligned[i] and close[i] > ema50_1d_aligned[i] and range_filter:
                signals[i] = 0.25
                position = 1
            # Bearish fractal (support break) + below trend -> short
            elif bearish_fractal_aligned[i] and close[i] < ema50_1d_aligned[i] and range_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on bearish fractal or trend reversal
            if bearish_fractal_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on bullish fractal or trend reversal
            if bullish_fractal_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals