#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band squeeze breakout with 12-hour volume confirmation
# In low volatility (BB width < 20th percentile), wait for expansion + volume spike to catch breakouts
# Works in both bull and bear markets by capturing volatility expansion moves
# Target: 20-35 trades/year to minimize fee drag while capturing high-conviction breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Bollinger Bands (20, 2.0) for squeeze detection ===
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + bb_std * std_20
    bb_lower = sma_20 - bb_std * std_20
    bb_width = bb_upper - bb_lower
    
    # === 12h Volume (for confirmation) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Volume moving average (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # === Bollinger Band width percentile (20-period lookback) ===
    # Calculate 20th percentile of BB width over last 20 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=20, min_periods=20).quantile(0.20).values
    
    # Align 12h data to 4h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_12h, bb_width_pct)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_pct_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volatility squeeze: current BB width < 20th percentile (low volatility)
        is_squeeze = bb_width[i] < bb_width_pct_aligned[i]
        
        # Volume spike: current 12h volume > 1.5x 20-period average
        vol_spike = volume_12h_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # Breakout detection: price breaks above upper BB or below lower BB
        breakout_up = close[i] > bb_upper[i-1]  # Break above previous period's upper BB
        breakout_down = close[i] < bb_lower[i-1]  # Break below previous period's lower BB
        
        # Entry logic: only enter when flat
        if position == 0:
            # Wait for volatility squeeze to end (expansion) + volume spike + breakout
            if not is_squeeze and vol_spike:
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: return to mean (middle of Bollinger Bands)
        elif position == 1:
            # Exit long when price returns to middle of Bollinger Bands
            if close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle of Bollinger Bands
            if close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4b_BBSqueeze_VolumeBreakout"
timeframe = "4h"
leverage = 1.0