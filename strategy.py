#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        ema_series = pd.Series(close_1d)
        ema50_1d = ema_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to daily timeframe (already aligned)
    ema50_1d_aligned = ema50_1d  # No alignment needed for same timeframe
    
    # Calculate daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d_arr[0]], close_1d_arr[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d_arr[0]], close_1d_arr[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to daily timeframe
    atr_1d_aligned = atr_1d  # No alignment needed for same timeframe
    
    # Calculate daily volume moving average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        volume_series = pd.Series(volume_1d)
        volume_ma_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume MA to daily timeframe
    volume_ma_1d_aligned = volume_ma_1d  # No alignment needed for same timeframe
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d_aligned[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 50% of 20-period MA)
        if volume[i] < 0.5 * volume_ma_1d_aligned[i]:
            signals[i] = 0.0
            continue
        
        # Get previous day's data for pivot calculation
        if i >= 1:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            
            # Calculate pivot points (standard formula)
            pivot = (prev_high + prev_low + prev_close) / 3.0
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            # For daily timeframe, pivot levels are constant for the day
            s3_level = s3
            r3_level = r3
            
            if position == 0:
                # Long: Price closes above S3 with volume and above EMA50 (bullish bias)
                if close[i] > s3_level and volume[i] > volume_ma_1d_aligned[i] and close[i] > ema50_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price closes below R3 with volume and below EMA50 (bearish bias)
                elif close[i] < r3_level and volume[i] > volume_ma_1d_aligned[i] and close[i] < ema50_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price closes below S3 or trend changes (price below EMA50)
                if close[i] < s3_level or close[i] < ema50_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price closes above R3 or trend changes (price above EMA50)
                if close[i] > r3_level or close[i] > ema50_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Pivot_S3R3_Rejection_Volume_EMA50_Filter_v1"
timeframe = "1d"
leverage = 1.0