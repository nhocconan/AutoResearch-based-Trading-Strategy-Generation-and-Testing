#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Breakout with 12h EMA Trend and Volume Confirmation
# Long when price breaks above upper Bollinger Band (20,2) with price > 12h EMA50 and volume > 1.5x average
# Short when price breaks below lower Bollinger Band with price < 12h EMA50 and volume > 1.5x average
# Exit on opposite band touch or volatility contraction (BB width < 0.5 * 20-period average width)
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets with strict entry criteria.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) - calculated on close
    bb_length = 20
    bb_mult = 2.0
    bb_basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_band = bb_basis + bb_dev
    lower_band = bb_basis - bb_dev
    
    # Bollinger Band Width for exit condition
    bb_width = (upper_band - lower_band) / bb_basis
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(bb_length, n):
        # Skip if required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_ma[i])):
            continue
        
        # Calculate 20-period average volume for confirmation
        vol_avg = np.mean(volume[max(0, i-19):i+1])
        
        # Long entry: price breaks above upper BB + price > 12h EMA50 + volume confirmation
        if (close[i] > upper_band[i] and 
            close[i] > ema_50_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower BB + price < 12h EMA50 + volume confirmation
        elif (close[i] < lower_band[i] and 
              close[i] < ema_50_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit conditions
        elif position == 1:
            # Exit on touch of lower band or volatility contraction
            if close[i] < lower_band[i] or bb_width[i] < 0.5 * bb_width_ma[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit on touch of upper band or volatility contraction
            if close[i] > upper_band[i] or bb_width[i] < 0.5 * bb_width_ma[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0