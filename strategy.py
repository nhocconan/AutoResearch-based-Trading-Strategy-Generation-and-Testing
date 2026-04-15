#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index + 1d RSI + 1d Volume Spike
# Uses Choppiness Index to detect ranging markets (CHOP > 61.8) for mean reversion,
# RSI(14) for overbought/oversold signals, and volume spike for confirmation.
# Works in both bull and bear markets by adapting to market regime - 
# in ranging markets, mean reversion at RSI extremes with volume confirmation.
# Target: 50-100 total trades over 4 years (12-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Choppiness Index (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros(len(close_arr))
        tr = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]),
                       abs(low_arr[i] - close_arr[i-1]))
        # Smooth TR using Wilder's smoothing (equivalent to RMA)
        atr[period-1] = np.sum(tr[1:period]) / period
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Calculate highest high and lowest low over period
        highest_high = np.zeros(len(close_arr))
        lowest_low = np.zeros(len(close_arr))
        for i in range(period-1, len(close_arr)):
            highest_high[i] = np.max(high_arr[i-period+1:i+1])
            lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        range_hl[range_hl == 0] = 1e-10
        
        chop = 100 * np.log10(atr * period / range_hl) / np.log10(period)
        return chop
    
    chop = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Calculate RSI (14-period)
    def calculate_rsi(close_arr, period=14):
        delta = np.diff(close_arr)
        delta = np.insert(delta, 0, 0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close_arr))
        avg_loss = np.zeros(len(close_arr))
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close_arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close_1d, 14)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe (prices are already 1d)
    chop_aligned = chop
    rsi_aligned = rsi
    vol_avg_aligned = vol_avg
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Choppy market (CHOP > 61.8) + RSI oversold (< 30) + volume spike
        if (chop_aligned[i] > 61.8 and 
            rsi_aligned[i] < 30 and 
            volume_1d[i] > 1.5 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Choppy market (CHOP > 61.8) + RSI overbought (> 70) + volume spike
        elif (chop_aligned[i] > 61.8 and 
              rsi_aligned[i] > 70 and 
              volume_1d[i] > 1.5 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral zone (40-60) or chop market ends
        elif position == 1 and (rsi_aligned[i] > 40 or chop_aligned[i] <= 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned[i] < 60 or chop_aligned[i] <= 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Chop_RSI_Volume_MeanReversion"
timeframe = "1d"
leverage = 1.0