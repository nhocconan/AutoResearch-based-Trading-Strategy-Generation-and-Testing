#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + volume spike + chop regime filter
# - TRIX(12): Triple Exponential Moving Average momentum oscillator
# - Long when TRIX crosses above 0 + volume > 2x 20-period average + chop > 61.8 (ranging market)
# - Short when TRIX crosses below 0 + volume > 2x 20-period average + chop > 61.8 (ranging market)
# - Chop > 61.8 indicates ranging conditions where mean reversion works best
# - Volume spike confirms momentum behind the TRIX crossover
# - Works in both bull (TRIX up with volume) and bear (TRIX down with volume) markets
# - Discrete position sizing ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits for 4h

name = "4h_trix_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate TRIX (12-period)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    trix_values = trix.values
    
    # Calculate TRIX crossover signals
    trix_cross_up = (trix_values > 0) & (np.roll(trix_values, 1) <= 0)
    trix_cross_down = (trix_values < 0) & (np.roll(trix_values, 1) >= 0)
    
    # Calculate Chopiness Index (14-period)
    atr_series = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1))))
    atr_sum = atr_series.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    chop_threshold = 61.8  # Above this = ranging market
    
    # Calculate volume spike (2x 20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_sma_20
    
    for i in range(14, n):  # Start after chop warmup period
        # Skip if any required data is invalid
        if (np.isnan(trix_values[i]) or np.isnan(chop_values[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: TRIX cross up + volume spike + chop > 61.8 (ranging)
        if trix_cross_up[i] and volume_spike[i] and chop_values[i] > chop_threshold:
            enter_long = True
        
        # Short: TRIX cross down + volume spike + chop > 61.8 (ranging)
        if trix_cross_down[i] and volume_spike[i] and chop_values[i] > chop_threshold:
            enter_short = True
        
        # Exit conditions: opposite TRIX crossover
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on TRIX cross down
            exit_long = trix_cross_down[i]
        elif position == -1:
            # Exit short on TRIX cross up
            exit_short = trix_cross_up[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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