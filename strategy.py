#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_cci_extreme_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate 14-day CCI on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # 14-period SMA of TP
    sma_tp = pd.Series(tp_1w).rolling(window=14, min_periods=14).mean().values
    
    # Mean deviation
    mad = pd.Series(tp_1w).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # CCI calculation
    cci_1w = (tp_1w - sma_tp) / (0.015 * mad)
    cci_1w[np.isnan(mad) | (mad == 0)] = 0
    
    # Align CCI to daily timeframe
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_1w)
    
    # Volume confirmation: volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Bands for exit (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    
    for i in range(300, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            signals[i] = 0.0
            continue
        
        cci_value = cci_1w_aligned[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Extreme CCI conditions
        cci_overbought = cci_value > 100
        cci_oversold = cci_value < -100
        
        # Entry signals
        long_signal = cci_oversold and volume_confirmed
        short_signal = cci_overbought and volume_confirmed
        
        # Exit when price returns to Bollinger Band middle (20-day SMA)
        exit_long = position == 1 and close[i] >= sma_20[i]
        exit_short = position == -1 and close[i] <= sma_20[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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

# Hypothesis: CCI extreme reversion on weekly timeframe with daily execution.
# Enters long when weekly CCI < -100 (extreme oversold) with volume confirmation (>2x avg volume).
# Enters short when weekly CCI > 100 (extreme overbought) with volume confirmation.
# Exits when price returns to the 20-day SMA (middle of Bollinger Bands).
# Works in both bull and bear markets by fading extreme weekly momentum.
# Weekly timeframe reduces noise, daily execution provides timely entry.
# Volume confirmation ensures institutional participation.
# Target: 15-30 trades/year to minimize fee drag in ranging markets.