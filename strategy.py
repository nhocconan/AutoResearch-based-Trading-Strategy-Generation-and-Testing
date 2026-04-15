#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze with 12h EMA Trend and Volume Spike
# Uses Bollinger Band width contraction (squeeze) on 4h to identify low volatility periods,
# then trades breakouts in the direction of 12h EMA trend, confirmed by volume spike.
# Works in both bull and bear markets by following the trend filter.
# Target: 50-150 total trades to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    bb_width = (upper - lower) / sma  # Normalized width
    
    # Bollinger Band Squeeze: width below 20-period mean width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Calculate EMA (50) on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: volume > 2x 20-period median volume
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    volume_spike = volume > (2 * vol_median)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if required data is not available
        if (np.isnan(sma[i]) or np.isnan(std[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i])):
            continue
        
        # Long entry: Bollinger breakout up + squeeze release + EMA uptrend + volume spike
        if (close[i] > upper[i] and
            not squeeze[i-1] and  # Squeeze released (width expanding)
            close[i-1] <= upper[i-1] and  # Was inside or below band
            ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and  # EMA rising
            volume_spike[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bollinger breakout down + squeeze release + EMA downtrend + volume spike
        elif (close[i] < lower[i] and
              not squeeze[i-1] and  # Squeeze released
              close[i-1] >= lower[i-1] and  # Was inside or above band
              ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and  # EMA falling
              volume_spike[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Bollinger band touch or squeeze re-formation
        elif position == 1 and (close[i] < sma[i] or squeeze[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > sma[i] or squeeze[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Squeeze_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0