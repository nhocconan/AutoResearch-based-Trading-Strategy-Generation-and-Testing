#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal + Weekly EMA Trend Filter + Volume Spike
# Uses weekly EMA(34) to determine trend direction (bullish/bearish).
# In bullish weekly trend: long on bullish fractal break above prior high with volume spike.
# In bearish weekly trend: short on bearish fractal break below prior low with volume spike.
# In neutral weekly trend: no trades to avoid chop.
# Volume confirmation: volume > 1.5x 20-period average to filter weak moves.
# Target: 15-30 trades/year per symbol to stay within frequency limits.
name = "12h_WilliamsFractal_WeeklyEMA_Trend_Volume"
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
    
    # Get weekly data for EMA trend and fractals
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Williams Fractals (5-bar: bar is highest/lowest of 2 bars on each side)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bullish_fractal = np.zeros(len(high_1w), dtype=bool)
    bearish_fractal = np.zeros(len(low_1w), dtype=bool)
    
    for i in range(2, len(high_1w) - 2):
        # Bullish fractal: lowest low of 5 bars
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = True
        # Bearish fractal: highest high of 5 bars
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = True
    
    # Get daily data for reference levels (prior day high/low)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Prior day high/low for breakout confirmation
    prior_high_1d = np.roll(high_1d, 1)
    prior_low_1d = np.roll(low_1d, 1)
    prior_high_1d[0] = np.nan  # First day has no prior
    prior_low_1d[0] = np.nan
    
    # Align indicators to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal.astype(float), additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal.astype(float), additional_delay_bars=2)
    prior_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_high_1d)
    prior_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_low_1d)
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20)  # Ensure EMA(34) + 2 delay, fractal + 2 delay, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(prior_high_1d_aligned[i]) or
            np.isnan(prior_low_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_1w_aligned[i]
        bullish_fract = bullish_fractal_aligned[i] > 0.5
        bearish_fract = bearish_fractal_aligned[i] > 0.5
        prior_high = prior_high_1d_aligned[i]
        prior_low = prior_low_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Weekly trend determination (using prior weekly close to avoid look-ahead)
        weekly_close = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
        if np.isnan(weekly_close_aligned[i]):
            signals[i] = 0.0
            continue
        prev_weekly_close = np.roll(weekly_close_aligned, 1)[i]
        prev_weekly_close = np.nan if i == 0 else prev_weekly_close
        if np.isnan(prev_weekly_close):
            signals[i] = 0.0
            continue
        weekly_trend_up = weekly_close_aligned[i] > ema_trend
        weekly_trend_down = weekly_close_aligned[i] < ema_trend
        
        if position == 0:
            # Determine entry based on weekly trend
            if weekly_trend_up and bullish_fract and volume_confirmed:
                # Bullish weekly trend: long on bullish fractal break above prior day high
                if price > prior_high:
                    signals[i] = 0.25
                    position = 1
            elif weekly_trend_down and bearish_fract and volume_confirmed:
                # Bearish weekly trend: short on bearish fractal break below prior day low
                if price < prior_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below prior day low or weekly EMA
            if price < prior_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above prior day high or weekly EMA
            if price > prior_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals