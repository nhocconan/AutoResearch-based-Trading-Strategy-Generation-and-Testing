#!/usr/bin/env python3
"""
4h_cci_trend_reversal_v2
Hypothesis: Improve upon previous CCI strategy by tightening entry conditions and adding ATR-based volatility filter to reduce false signals. Enter long when CCI crosses above -80 with above-average volume and price above 20-period EMA, short when CCI crosses below +80 with above-average volume and price below 20-period EMA. Exit when CCI crosses zero. Use 1d ADX > 25 to ensure trending market. Designed for 20-40 trades/year to minimize fee drag while capturing momentum reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_trend_reversal_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h CCI (20-period)
    if len(close) < 20:
        return np.zeros(n)
    
    # Typical Price
    tp = (high + low + close) / 3.0
    
    # Moving Average of TP
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    
    # Mean Deviation
    md = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # CCI
    cci = (tp - ma_tp) / (0.015 * md)
    
    # Calculate 20-period EMA for trend filter
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX for trend filter (avoid ranging markets)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: smoothed = prev_smoothed * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align indicators to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr[i] > 0.5 * np.nanmedian(atr[max(0, i-50):i+1]) if i >= 50 else True
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: CCI crosses below zero (momentum exhaustion)
            if cci[i] < 0 and cci[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above zero (momentum exhaustion)
            if cci[i] > 0 and cci[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and vol_filter and trend_filter:
                # Long: CCI crosses above -80 with price above EMA20
                if cci[i] > -80 and cci[i-1] <= -80 and close[i] > ema_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: CCI crosses below +80 with price below EMA20
                elif cci[i] < 80 and cci[i-1] >= 80 and close[i] < ema_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals