# 12h_Camarilla_Pivot_Trend_Filter
# Hypothesis: Combines daily Camarilla pivot levels with trend filters (EMA50/200) and volume spikes
# Works in bull markets via breakout above pivot resistance and in bear via breakdown below pivot support
# Daily Camarilla provides institutional reference points, EMA filters avoid counter-trend trades
# Volume spike confirms institutional participation, reducing false breakouts
# Target: 12h timeframe with low trade frequency (<30/year) to minimize fee drag
# Uses 1d HTF for pivots and EMAs, avoiding look-ahead bias with proper alignment

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and EMAs
    daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: R4 = Close + (High-Low)*1.1/2, R3 = Close + (High-Low)*1.1/4, etc.
    # Using previous day's data to avoid look-ahead
    daily_close = daily['close'].values
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    
    # Shift by 1 to use previous day's data
    prev_close = np.concatenate([[np.nan], daily_close[:-1]])
    prev_high = np.concatenate([[np.nan], daily_high[:-1]])
    prev_low = np.concatenate([[np.nan], daily_low[:-1]])
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    r4 = prev_close + rang * 1.1 / 2
    r3 = prev_close + rang * 1.1 / 4
    r2 = prev_close + rang * 1.1 / 6
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12
    s2 = prev_close - rang * 1.1 / 6
    s3 = prev_close - rang * 1.1 / 4
    s4 = prev_close - rang * 1.1 / 2
    
    # Align to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, daily, r4)
    r3_aligned = align_htf_to_ltf(prices, daily, r3)
    r2_aligned = align_htf_to_ltf(prices, daily, r2)
    r1_aligned = align_htf_to_ltf(prices, daily, r1)
    s1_aligned = align_htf_to_ltf(prices, daily, s1)
    s2_aligned = align_htf_to_ltf(prices, daily, s2)
    s3_aligned = align_htf_to_ltf(prices, daily, s3)
    s4_aligned = align_htf_to_ltf(prices, daily, s4)
    
    # Calculate EMAs for trend filter
    daily_ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_200 = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, daily, daily_ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, daily, daily_ema_200)
    
    # Volume spike detection
    vol_ema_20 = pd.Series(daily['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, daily, vol_ema_20)
    vol_spike = volume > (2.0 * vol_ema_20_aligned)  # 2x average volume
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            continue
        
        # Long conditions:
        # 1. Price above R3 (strong resistance turned support)
        # 2. Price above both EMA50 and EMA200 (bullish trend)
        # 3. Volume spike (institutional participation)
        if (close[i] > r3_aligned[i] and 
            close[i] > ema_50_aligned[i] and 
            close[i] > ema_200_aligned[i] and 
            vol_spike[i]):
            signals[i] = 0.25
        
        # Short conditions:
        # 1. Price below S3 (strong support turned resistance)
        # 2. Price below both EMA50 and EMA200 (bearish trend)
        # 3. Volume spike (institutional participation)
        elif (close[i] < s3_aligned[i] and 
              close[i] < ema_50_aligned[i] and 
              close[i] < ema_200_aligned[i] and 
              vol_spike[i]):
            signals[i] = -0.25
        
        # Exit conditions:
        # Long exit: price falls below R1 or EMA50
        elif signals[i-1] > 0 and (close[i] < r1_aligned[i] or close[i] < ema_50_aligned[i]):
            signals[i] = 0.0
        
        # Short exit: price rises above S1 or EMA50
        elif signals[i-1] < 0 and (close[i] > s1_aligned[i] or close[i] > ema_50_aligned[i]):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Camarilla_Pivot_Trend_Filter"
timeframe = "12h"
leverage = 1.0