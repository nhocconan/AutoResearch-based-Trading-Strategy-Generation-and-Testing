#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour trend following using 12-hour Supertrend with 1-day ATR filter
# Supertrend on 12h identifies trend direction using ATR-based dynamic bands
# Only take trades when 1-day ATR ratio (ATR14/ATR50) > 0.8 to avoid low-volatility chop
# Volume confirmation (>1.3x 20-bar average) ensures institutional participation
# Designed for 6h timeframe to target 60-120 total trades over 4 years (15-30/year)
# Works in bull markets via trend continuation, avoids whipsaws in bear via volatility filter

name = "6h_Supertrend12h_ATRRatio1d_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 10 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Supertrend on 12h timeframe
    def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # ATR using Wilder's smoothing
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])  # Skip first NaN
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Basic Upper and Lower Bands
        hl2 = (high + low) / 2
        upper_basic = hl2 + multiplier * atr
        lower_basic = hl2 - multiplier * atr
        
        # Final Upper and Lower Bands
        upper_final = np.full_like(upper_basic, np.nan)
        lower_final = np.full_like(lower_basic, np.nan)
        
        for i in range(len(close)):
            if np.isnan(atr[i]) or i == 0:
                upper_final[i] = upper_basic[i]
                lower_final[i] = lower_basic[i]
            else:
                if upper_basic[i] <= upper_final[i-1] or close[i-1] > upper_final[i-1]:
                    upper_final[i] = upper_basic[i]
                else:
                    upper_final[i] = upper_final[i-1]
                    
                if lower_basic[i] >= lower_final[i-1] or close[i-1] < lower_final[i-1]:
                    lower_final[i] = lower_basic[i]
                else:
                    lower_final[i] = lower_final[i-1]
        
        # Supertrend
        supertrend = np.full_like(close, np.nan)
        for i in range(len(close)):
            if np.isnan(atr[i]) or i == 0:
                supertrend[i] = upper_final[i]
            else:
                if supertrend[i-1] == upper_final[i-1] and close[i] <= upper_final[i]:
                    supertrend[i] = lower_final[i]
                elif supertrend[i-1] == lower_final[i-1] and close[i] >= lower_final[i]:
                    supertrend[i] = upper_final[i]
                elif supertrend[i-1] == upper_final[i-1]:
                    supertrend[i] = upper_final[i]
                else:
                    supertrend[i] = lower_final[i]
        
        # Trend direction: 1 for uptrend (price above supertrend), -1 for downtrend
        trend = np.where(close > supertrend, 1, -1)
        return trend, upper_final, lower_final
    
    trend_12h, upper_band_12h, lower_band_12h = calculate_supertrend(high_12h, low_12h, close_12h, 10, 3.0)
    
    # Calculate ATR ratio on 1d: ATR14/ATR50 to filter for sufficient volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # ATR with Wilder's smoothing
    def atr_wilder(data, period):
        atr = np.full_like(data, np.nan)
        if len(data) >= period:
            atr[period-1] = np.nanmean(data[1:period])  # Skip first NaN
            for i in range(period, len(data)):
                atr[i] = (atr[i-1] * (period-1) + data[i]) / period
        return atr
    
    atr_14_1d = atr_wilder(tr_1d, 14)
    atr_50_1d = atr_wilder(tr_1d, 50)
    atr_ratio_1d = np.where(atr_50_1d != 0, atr_14_1d / atr_50_1d, 0)
    
    # Volume confirmation (>1.3x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: uptrend on 12h AND sufficient volatility (ATR ratio > 0.8) AND volume confirmation
            if (trend_12h_aligned[i] == 1 and atr_ratio_1d_aligned[i] > 0.8 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend on 12h AND sufficient volatility AND volume confirmation
            elif (trend_12h_aligned[i] == -1 and atr_ratio_1d_aligned[i] > 0.8 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend changes to downtrend on 12h
            if trend_12h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend changes to uptrend on 12h
            if trend_12h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals