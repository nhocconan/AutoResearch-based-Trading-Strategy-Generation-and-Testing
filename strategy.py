#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter (10 EMA) and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (price > 10 EMA),
# we fade extreme readings: buy when %R < -80 (oversold), sell when %R > -20 (overbought).
# Volume confirmation ensures participation. Designed for mean reversion within trends,
# working in both bull and bear markets by aligning with trend direction.
# Targets 20-40 trades/year with strict entry conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 10-period EMA on daily close
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    
    # Calculate Williams %R on 4h data (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)  # Handle division by zero
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for Williams %R
        # Skip if data not ready
        if (np.isnan(ema_10_aligned[i]) or 
            np.isnan(wr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr_val = wr[i]
        ema10 = ema_10_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Only trade in direction of daily trend: price above/below 10 EMA
            if price > ema10:  # Uptrend bias
                if wr_val < -80 and vol_confirmed:  # Oversold with volume
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend bias
                if wr_val > -20 and vol_confirmed:  # Overbought with volume
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to neutral zone or trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when overbought or trend turns down
                if wr_val > -20 or price < ema10:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when oversold or trend turns up
                if wr_val < -80 or price > ema10:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0