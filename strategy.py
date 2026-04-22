#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w trend filter and volume confirmation
# Long when price breaks above bullish fractal high with 1w uptrend and volume spike
# Short when price breaks below bearish fractal low with 1w downtrend and volume spike
# Uses Williams Fractals (5-bar pattern) for natural support/resistance levels
# Weekly trend filter reduces whipsaws, volume confirmation ensures momentum
# Designed for 1d timeframe targeting 8-20 trades/year per symbol
# Works in bull markets (breakouts with trend) and bear markets (fades from fractal levels)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Load 1d data for fractal calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals (5-bar pattern)
    # Bearish fractal: high[n-2] is highest of high[n-4:n]
    # Bullish fractal: low[n-2] is lowest of low[n-4:n]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] == np.max(high_1d[i-2:i+3]) and 
            high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and
            high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        if (low_1d[i] == np.min(low_1d[i-2:i+3]) and 
            low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and
            low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align fractals to 1d timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1w EMA(34) for higher timeframe trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike filter (20-period on 1d data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above bullish fractal + 1w uptrend + volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal + 1w downtrend + volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite fractal level or trend reversal
            if position == 1:
                # Exit on price below bearish fractal or trend reversal
                if (close[i] < bearish_fractal_aligned[i] or 
                    close[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above bullish fractal or trend reversal
                if (close[i] > bullish_fractal_aligned[i] or 
                    close[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0