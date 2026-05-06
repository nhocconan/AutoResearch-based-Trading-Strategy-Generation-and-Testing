#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 200 EMA with 1h trend filter and volume confirmation
# Uses 200 EMA on 1d timeframe for long-term trend direction
# Requires price to be above/below 200 EMA with 1h ADX(25) > 20 for trend confirmation
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in both bull/bear: captures strong trends, avoids false signals in consolidation

name = "1d_EMA200_1hADX20_VolumeConfirm_v1"
timeframe = "1d"
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
    df_1d = get_htf_data(prices, '1d')
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1d) < 200 or len(df_1h) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 200 EMA on 1d timeframe
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1h ADX(20) trend filter
    tr1 = np.abs(high_1h[1:] - low_1h[1:])
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    dm_plus = np.where((high_1h[1:] - high_1h[:-1]) > (low_1h[:-1] - low_1h[1:]), 
                       np.maximum(high_1h[1:] - high_1h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    dm_minus = np.where((low_1h[:-1] - low_1h[1:]) > (high_1h[1:] - high_1h[:-1]), 
                        np.maximum(low_1h[:-1] - low_1h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1h = wilder_smooth(tr, 20)
    dm_plus_smooth = wilder_smooth(dm_plus, 20)
    dm_minus_smooth = wilder_smooth(dm_minus, 20)
    
    di_plus = np.where(atr_1h != 0, 100 * dm_plus_smooth / atr_1h, 0)
    di_minus = np.where(atr_1h != 0, 100 * dm_minus_smooth / atr_1h, 0)
    
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    adx_1h = wilder_smooth(dx, 20)
    
    # Calculate ATR(14) for 1d timeframe (for stoploss)
    tr1_1d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe (primary)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(200, n):  # Start after EMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(adx_1h_aligned[i]) or 
            np.isnan(atr_1d[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long entry: price above 200 EMA AND trending market (ADX > 20) AND volume confirmation
            if (close[i] > ema_200_1d_aligned[i] and 
                adx_1h_aligned[i] > 20 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short entry: price below 200 EMA AND trending market (ADX > 20) AND volume confirmation
            elif (close[i] < ema_200_1d_aligned[i] and 
                  adx_1h_aligned[i] > 20 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price closes below the 200 EMA
            if close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price closes above the 200 EMA
            if close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals