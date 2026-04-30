#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1w EMA34 trend filter and volume confirmation
# Uses Williams Fractals from 1w for structure (proven edge on BTC/ETH)
# Only trade bullish fractal breaks above 1w EMA34 (uptrend) or bearish fractal breaks below 1w EMA34 (downtrend)
# Volume spike (2.0x 20-period average) confirms institutional participation
# 1w EMA34 provides smoother trend than shorter EMAs, reducing whipsaw in ranging markets
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by following the 1w EMA34 trend direction.

name = "12h_WilliamsFractal_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Fractals on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Initialize fractal arrays
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    # Williams Fractals: need 5 bars (2 left, 2 right)
    for i in range(2, len(high_1w) - 2):
        # Bearish fractal: high[i] is highest among 5 bars
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        
        # Bullish fractal: low[i] is lowest among 5 bars
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Align fractals to 12h timeframe with 2-bar extra delay for confirmation
    # Williams fractals need 2 extra 1w bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA calculation
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_bearish_fractal = bearish_fractal_aligned[i]
        curr_bullish_fractal = bullish_fractal_aligned[i]
        
        # Skip if fractal values are not available (NaN)
        if np.isnan(curr_ema_34_1w):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: bullish fractal break AND above 1w EMA34 (uptrend)
                if not np.isnan(curr_bullish_fractal) and curr_close > curr_bullish_fractal and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: bearish fractal break AND below 1w EMA34 (downtrend)
                elif not np.isnan(curr_bearish_fractal) and curr_close < curr_bearish_fractal and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below bullish fractal or below 1w EMA34
            exit_condition = False
            if not np.isnan(curr_bullish_fractal) and curr_close < curr_bullish_fractal:
                exit_condition = True
            elif curr_close < curr_ema_34_1w:
                exit_condition = True
                
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above bearish fractal or above 1w EMA34
            exit_condition = False
            if not np.isnan(curr_bearish_fractal) and curr_close > curr_bearish_fractal:
                exit_condition = True
            elif curr_close > curr_ema_34_1w:
                exit_condition = True
                
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals