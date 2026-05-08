#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal + Volume Spike + Weekly Trend Filter
# Uses Williams Fractals (5-bar) for swing point detection, volume spike for confirmation,
# and weekly EMA for trend filter. Long on bullish fractal breakout above resistance with volume spike in uptrend.
# Short on bearish fractal breakdown below support with volume spike in downtrend.
# Exit on opposite fractal signal or trend reversal.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull (trend follow breakouts) and bear (mean reversion via fractal reversals in range).

name = "1d_WilliamsFractal_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams Fractals (5-bar: 2 left, 2 right)
    # Bearish fractal: high[n-2] < high[n] > high[n+2] and high[n-1] < high[n] > high[n+1]
    # Bullish fractal: low[n-2] > low[n] < low[n+2] and low[n-1] > low[n] < low[n+1]
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (high[i-2] < high[i] and high[i-1] < high[i] and 
            high[i+1] < high[i] and high[i+2] < high[i]):
            bearish_fractal[i] = True
        if (low[i-2] > low[i] and low[i-1] > low[i] and 
            low[i+1] > low[i] and low[i+2] > low[i]):
            bullish_fractal[i] = True
    
    # Volume spike: volume > 1.5 * 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Weekly EMA(21) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_1w_up = ema_21_1w[1:] > ema_21_1w[:-1]
    trend_1w_up = np.concatenate([[False], trend_21_1w])
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    
    # Resistance and support levels from recent fractals
    # Resistance: highest bearish fractal high in last 20 periods
    # Support: lowest bullish fractal low in last 20 periods
    resistance = np.full(n, np.nan)
    support = np.full(n, np.nan)
    
    for i in range(20, n):
        # Look back 20 periods for fractals
        lookback_high = high[max(0, i-20):i]
        lookback_low = low[max(0, i-20):i]
        bearish_in_lookback = bearish_fractal[max(0, i-20):i]
        bullish_in_lookback = bullish_fractal[max(0, i-20):i]
        
        if np.any(bearish_in_lookback):
            # Get highest high where bearish fractal occurred
            idx = np.where(bearish_in_lookback)[0]
            resistance[i] = np.max(lookback_high[idx])
        if np.any(bullish_in_lookback):
            # Get lowest low where bullish fractal occurred
            idx = np.where(bullish_in_lookback)[0]
            support[i] = np.min(lookback_low[idx])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(resistance[i]) or np.isnan(support[i]) or 
            np.isnan(trend_1w_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: bullish fractal breakout above resistance with volume spike in uptrend
            if (bullish_fractal[i] and 
                close[i] > resistance[i] and 
                volume_spike[i] and 
                trend_1w_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish fractal breakdown below support with volume spike in downtrend
            elif (bearish_fractal[i] and 
                  close[i] < support[i] and 
                  volume_spike[i] and 
                  not trend_1w_up_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish fractal breakdown or trend reversal
            if bearish_fractal[i] and close[i] < support[i]:
                signals[i] = 0.0
                position = 0
            elif not trend_1w_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish fractal breakout or trend reversal
            if bullish_fractal[i] and close[i] > resistance[i]:
                signals[i] = 0.0
                position = 0
            elif trend_1w_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals