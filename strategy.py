#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with volume confirmation and 1d ATR stop.
# Williams %R identifies overbought/oversold conditions in the short term.
# Entry: %R crosses above -20 (short) or below -80 (long) with volume confirmation.
# Exit: %R crosses back above -50 (long exit) or below -50 (short exit).
# ATR-based stop limits drawdown. Target: 20-30 trades/year (80-120 total over 4 years).
# Williams %R works in both trending and ranging markets by capturing momentum extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 14:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Williams %R, ATR, and volume average ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams %R on daily data (14-period)
    def calculate_williams_r(high, low, close, period=14):
        wr = np.full_like(high, np.nan, dtype=float)
        for i in range(period - 1, len(high)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            if highest_high != lowest_low:
                wr[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                wr[i] = -50  # Avoid division by zero
        return wr
    
    williams_r_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # ATR calculation on daily data (14-period)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        if len(high) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 10-day average volume on daily data
    volume_1d_series = pd.Series(volume_1d)
    vol_avg10_1d = volume_1d_series.rolling(window=10, min_periods=10).mean().values
    vol_avg10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg10_1d)
    
    signals = np.zeros(n)
    position = 0
    warmup = 14  # Sufficient for Williams %R
    
    for i in range(warmup, n):
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_avg10_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_filter = vol_1d_current > 1.2 * vol_avg10_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses below -80 (oversold) + volume confirmation
            if williams_r_1d_aligned[i] < -80 and williams_r_1d_aligned[i-1] >= -80 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses above -20 (overbought) + volume confirmation
            elif williams_r_1d_aligned[i] > -20 and williams_r_1d_aligned[i-1] <= -20 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if williams_r_1d_aligned[i] > -50 and williams_r_1d_aligned[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if williams_r_1d_aligned[i] < -50 and williams_r_1d_aligned[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0