#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Williams Fractal breakout and volume confirmation.
Trade breakouts of weekly Williams Fractal levels with volume spike (>2x 20-period average).
Use 1d ADX > 20 to filter for trending markets.
In trending markets: buy breakouts above weekly bearish fractal, sell breakdowns below weekly bullish fractal.
In ranging markets (ADX <= 20): fade the fractal levels with smaller position.
Position sizing: 0.30 for breakouts in trend, 0.15 for mean reversion in range, 0 for exit.
Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Get 1d data for ADX and volume MA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ADX (14) on 1d
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Williams Fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 after)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Align 1d indicators to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction and strength
        uptrend = plus_di_aligned[i] > minus_di_aligned[i]
        downtrend = plus_di_aligned[i] < minus_di_aligned[i]
        strong_trend = adx_aligned[i] > 20
        
        if position == 0:
            # In strong trend: trade breakouts of weekly fractals
            if strong_trend:
                # Long: price breaks above weekly bearish fractal with volume spike
                if (close[i] > bearish_fractal_aligned[i] and 
                    volume[i] > vol_ma_20_aligned[i] * 2.0):
                    signals[i] = 0.30
                    position = 1
                # Short: price breaks below weekly bullish fractal with volume spike
                elif (close[i] < bullish_fractal_aligned[i] and 
                      volume[i] > vol_ma_20_aligned[i] * 2.0):
                    signals[i] = -0.30
                    position = -1
            # In ranging market: fade the fractal levels (mean reversion)
            else:
                # Long: price near weekly bullish fractal (support) with volume
                if (close[i] <= bullish_fractal_aligned[i] * 1.02 and  # within 2% above
                    volume[i] > vol_ma_20_aligned[i] * 1.5):
                    signals[i] = 0.15
                    position = 1
                # Short: price near weekly bearish fractal (resistance) with volume
                elif (close[i] >= bearish_fractal_aligned[i] * 0.98 and  # within 2% below
                      volume[i] > vol_ma_20_aligned[i] * 1.5):
                    signals[i] = -0.15
                    position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly bullish fractal or ADX drops
            if close[i] < bullish_fractal_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if strong_trend else 0.15
        
        elif position == -1:
            # Exit short: price crosses above weekly bearish fractal or ADX drops
            if close[i] > bearish_fractal_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30 if strong_trend else -0.15
    
    return signals

name = "1d_WilliamsFractal_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0