#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_ma_cross_volume_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Weekly SMA(8) and SMA(21) for trend
    close_1w = df_1w['close'].values
    sma_8_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    sma_21_1w = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
    
    # Shift by 1 to use only completed weekly bars
    sma_8_1w = np.roll(sma_8_1w, 1)
    sma_21_1w = np.roll(sma_21_1w, 1)
    sma_8_1w[0] = np.nan
    sma_21_1w[0] = np.nan
    
    # Align weekly SMA to 6h timeframe
    sma_8_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_8_1w)
    sma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_21_1w)
    
    # Volume filter: volume > 1.8x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # EMA(50) on 6h for exit condition
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(sma_8_1w_aligned[i]) or np.isnan(sma_21_1w_aligned[i]) or
            np.isnan(vol_ma_50[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_50[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Weekly MA crossover signals
        bullish_cross = sma_8_1w_aligned[i] > sma_21_1w_aligned[i]
        bearish_cross = sma_8_1w_aligned[i] < sma_21_1w_aligned[i]
        
        # Entry conditions
        long_entry = bullish_cross and volume_confirmed
        short_entry = bearish_cross and volume_confirmed
        
        # Exit conditions: price crosses 50 EMA in opposite direction
        exit_long = position == 1 and price_close < ema_50[i]
        exit_short = position == -1 and price_close > ema_50[i]
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly MA crossover with volume filter on 6h timeframe.
# Uses weekly SMA(8)/SMA(21) crossovers to determine trend direction.
# Enters only when volume exceeds 1.8x 50-period average to avoid false signals.
# Exits when price crosses the 6h EMA(50) in the opposite direction.
# Designed for low trade frequency (target: 25-40 trades/year) to minimize fee drag.
# Works in both bull and bear markets by following the higher timeframe trend.