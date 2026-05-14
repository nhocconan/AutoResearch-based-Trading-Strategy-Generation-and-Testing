#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above bullish fractal and close > 1d EMA34 with volume > 1.5x average.
Short when price breaks below bearish fractal and close < 1d EMA34 with volume > 1.5x average.
Exit on opposite fractal break or trend reversal.
Williams Fractals provide swing high/low structure that works in both trending and ranging markets.
1d EMA34 filters medium-term trend, volume confirmation ensures breakout legitimacy.
Designed for 12h timeframe targeting 50-150 total trades over 4 years with low frequency to minimize fee drag.
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
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Load 1d data for Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Compute Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align fractals to 12h timeframe with additional delay for confirmation (2 bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > bullish_fractal_val and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below bearish fractal AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < bearish_fractal_val and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below bearish fractal OR trend reversal
                if (price < bearish_fractal_val or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above bullish fractal OR trend reversal
                if (price > bullish_fractal_val or price > ema34_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsFractal_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0