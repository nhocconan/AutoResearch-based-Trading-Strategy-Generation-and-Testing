#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot (R1/S1) breakout with 1-day EMA34 trend filter and volume confirmation.
Long when price breaks above R1 with EMA34 rising and volume > 1.5x average.
Short when price breaks below S1 with EMA34 falling and volume > 1.5x average.
Exit when price returns to the mean (Pivot point) or reverses at opposite level.
Camarilla levels provide precise intraday support/resistance; daily EMA filters trend; volume avoids false breakouts.
Designed for low trade frequency by requiring multiple confirmations. Works in both bull and bear markets by following daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Previous day's typical price (shift by 1)
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan  # First value invalid
    
    # Previous day's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    # R1 = close + (range * 1.1/12)
    # S1 = close - (range * 1.1/12)
    R1 = prev_typical + (prev_range * 1.1 / 12)
    S1 = prev_typical - (prev_range * 1.1 / 12)
    Pivot = prev_typical  # Central pivot point
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for volume MA
        # Skip if data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(Pivot[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, EMA34 rising, volume confirmation
            if (close[i] > R1[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, EMA34 falling, volume confirmation
            elif (close[i] < S1[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to pivot or breaks below S1 (reversal)
                if (close[i] <= Pivot[i] or 
                    close[i] < S1[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to pivot or breaks above R1 (reversal)
                if (close[i] >= Pivot[i] or 
                    close[i] > R1[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0