#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20-period) with volume confirmation and EMA trend filter.
# Donchian channels act as dynamic support/resistance; breakouts with volume indicate strong momentum.
# EMA(50) filters for trend alignment to avoid counter-trend entries.
# Designed for 4h timeframe to capture medium-term breakouts with low frequency.
# Entry: Long when close > upper Donchian band and volume > 1.5x average and close > EMA50;
#        Short when close < lower Donchian band and volume > 1.5x average and close < EMA50.
# Exit: Opposite Donchian band touch or EMA50 crossover.
# Uses strict conditions to limit trades (~20-40/year) and avoid overtrading.
# Works in bull markets via trend-following breakouts and in bear markets via short breakdowns.
name = "4h_Donchian_EMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with volume and trend alignment
            if (close[i] > donchian_high[i] and 
                volume_confirm[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and trend alignment
            elif (close[i] < donchian_low[i] and 
                  volume_confirm[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower Donchian or trend turns bearish
            if (close[i] < donchian_low[i]) or (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper Donchian or trend turns bullish
            if (close[i] > donchian_high[i]) or (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals