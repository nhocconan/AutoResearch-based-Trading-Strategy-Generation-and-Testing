#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Camarilla pivot reversal + volume confirmation
# - Uses 1d Camarilla pivot levels (R1, S1) for reversal signals
# - Long when price crosses below S1 with bullish rejection (close > open)
# - Short when price crosses above R1 with bearish rejection (close < open)
# - Volume confirmation: current 12h volume > 1.5x 20-period average
# - Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts
# - Designed to work in both bull and bear markets by capturing reversals at key levels
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "12h_Camarilla_Pivot_Reversal_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # Formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+C)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    rang = df_1d['high'] - df_1d['low']
    r1 = typical_price + rang * 1.1 / 12
    s1 = typical_price - rang * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume average (20-period)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (00:00-24:00 UTC - trade all hours for 12h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    # Always trade for 12h timeframe (no session filter needed)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x average
        volume_filter = vol_ma_12h[i] > 0 and volume[i] > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Look for long entry: price crosses below S1 with bullish rejection + uptrend + volume
            if (low[i] <= s1_aligned[i] and close[i] > open_price[i] and  # bullish rejection
                close[i] > ema_50_1d_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: price crosses above R1 with bearish rejection + downtrend + volume
            elif (high[i] >= r1_aligned[i] and close[i] < open_price[i] and  # bearish rejection
                  close[i] < ema_50_1d_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish rejection at R1 or trend reversal
            if (high[i] >= r1_aligned[i] and close[i] < open_price[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish rejection at S1 or trend reversal
            if (low[i] <= s1_aligned[i] and close[i] > open_price[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals