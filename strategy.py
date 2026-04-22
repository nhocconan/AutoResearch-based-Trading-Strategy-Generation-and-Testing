#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(21) trend filter and volume confirmation.
# Uses 12h Donchian for breakout levels, 12h EMA for trend direction, and volume spike for confirmation.
# Long when price breaks above upper Donchian in uptrend (close > 12h EMA21) with volume spike.
# Short when price breaks below lower Donchian in downtrend (close < 12h EMA21) with volume spike.
# Exit on opposite Donchian touch or trend reversal.
# Designed for 4h timeframe to target 20-50 trades/year per symbol.
# Works in bull/bear via trend filter + volatility-based breakout levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian and trend (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) for 12h: upper = max(high,20), lower = min(low,20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(21) for trend direction
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to 4h timeframe (waits for 12h bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend (close > EMA21) + volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_21_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + downtrend (close < EMA21) + volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_21_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on lower Donchian touch or trend reversal
                if (close[i] < donchian_lower_aligned[i] or close[i] < ema_21_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on upper Donchian touch or trend reversal
                if (close[i] > donchian_upper_aligned[i] or close[i] > ema_21_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0