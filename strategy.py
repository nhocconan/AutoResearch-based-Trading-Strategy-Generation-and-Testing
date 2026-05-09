#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level (R3/S3) bounce with 1d EMA34 trend filter and volume spike.
# Uses daily EMA34 for trend direction (price > EMA34 = bullish, < EMA34 = bearish),
# Camarilla R3/S3 levels for mean-reversion entries, and volume surge for confirmation.
# In bull trend: buy at S3 bounce; in bear trend: sell at R3 rejection.
# Designed for low trade frequency (<400 total) to minimize fee drag.
name = "4h_Camarilla_R3S3_Bounce_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA for daily timeframe
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels for each day (based on previous day OHLC)
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.2500)
    # R2 = close + ((high - low) * 1.1666)
    # R1 = close + ((high - low) * 1.0833)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.0833)
    # S2 = close - ((high - low) * 1.1666)
    # S3 = close - ((high - low) * 1.2500)
    # S4 = close - ((high - low) * 1.5000)
    # We use R3 and S3 for entries
    
    # We need previous day's OHLC to calculate today's Camarilla levels
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first day's previous values to current day's (no look-ahead)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    # Calculate Camarilla R3 and S3
    R3 = close_prev + ((high_prev - low_prev) * 1.2500)
    S3 = close_prev - ((high_prev - low_prev) * 1.2500)
    
    # Volume confirmation: volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have valid previous day data
        # Skip if required data unavailable
        if (np.isnan(ema34_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # In bull trend (price > EMA34): look for long at S3 bounce
            # In bear trend (price < EMA34): look for short at R3 rejection
            if close[i] > ema34_aligned[i]:  # Bull trend
                if price <= S3[i] * 1.001 and price >= S3[i] * 0.999:  # Near S3 level (within 0.1%)
                    if vol_confirm[i]:
                        signals[i] = 0.25
                        position = 1
            else:  # Bear trend
                if price <= R3[i] * 1.001 and price >= R3[i] * 0.999:  # Near R3 level (within 0.1%)
                    if vol_confirm[i]:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Exit long: price reaches R3 or trend changes to bear
            if price >= R3[i] * 0.999 or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S3 or trend changes to bull
            if price <= S3[i] * 1.001 or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals