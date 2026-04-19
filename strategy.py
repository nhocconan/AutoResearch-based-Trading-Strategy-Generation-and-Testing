#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with weekly trend filter and volume confirmation.
# Long when: price closes above upper BB(20,2) and weekly EMA(34) rising, with volume > 1.5x 20-day avg volume
# Short when: price closes below lower BB(20,2) and weekly EMA(34) falling, with volume > 1.5x 20-day avg volume
# Exit when price reverts to middle BB or volatility expands.
# Designed for ~15-25 trades/year per symbol to avoid fee drag.
name = "1d_BB_Breakout_WeeklyTrend_Volume"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_slope = ema_1w_34 - np.roll(ema_1w_34, 1)
    ema_1w_34_slope[0] = 0
    ema_1w_34_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34_slope)
    
    # Bollinger Bands (20,2) on daily
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_34_slope_aligned[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_slope = ema_1w_34_slope_aligned[i]
        vol_ratio = volume_ratio[i]
        
        if position == 0:
            # Long: price closes above upper BB, weekly trend up, volume confirmation
            if close[i] > upper_bb[i] and weekly_slope > 0 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower BB, weekly trend down, volume confirmation
            elif close[i] < lower_bb[i] and weekly_slope < 0 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle BB or volatility expands (volume drops)
            if close[i] < middle_bb[i] or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle BB or volatility expands (volume drops)
            if close[i] > middle_bb[i] or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals