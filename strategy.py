#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above latest bullish fractal in bull trend (close > 1w EMA50) with volume > 1.8x 20-period MA.
# Short when price breaks below latest bearish fractal in bear trend (close < 1w EMA50) with volume spike.
# Williams fractals require 2-bar confirmation delay on 1w timeframe. Uses discrete sizing (0.25) to minimize fees.
# Target: 50-150 total trades over 4 years = 12-37/year. Works in bull via breakouts, bear via shorting fractal breakdowns.

name = "6h_WilliamsFractal_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter and Williams fractals
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams fractals on 1w (requires 2-bar confirmation delay)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    n_1w = len(high_1w)
    bullish_fractal = np.full(n_1w, np.nan)
    bearish_fractal = np.full(n_1w, np.nan)
    
    for i in range(2, n_1w - 2):
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bullish_fractal[i] = high_1w[i]
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bearish_fractal[i] = low_1w[i]
    
    # Align fractals to 6h timeframe with 2-bar additional delay for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Fractal breakout conditions
        breakout_bull = close_val > bull_fractal
        breakout_bear = close_val < bear_fractal
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_bull and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_bear and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below bearish fractal OR trend reversal
            if close_val < bear_fractal or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above bullish fractal OR trend reversal
            if close_val > bull_fractal or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals