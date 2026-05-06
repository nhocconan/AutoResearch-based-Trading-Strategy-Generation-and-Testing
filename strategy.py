#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 12h trend filter and volume confirmation
# Uses Camarilla pivot levels from 1d timeframe for entry/exit signals
# Requires price to touch S1/R1 or S3/R3 levels with reversal confirmation
# Uses 12h EMA(50) for trend direction and 12h ADX(25) for trend strength filter
# Volume confirmation (>1.5x 20-bar average) ensures institutional participation
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)
# Works in both bull/bear: captures reversals at key levels with trend alignment

name = "4h_Camarilla_R1S1_12hEMA50_ADX25_Volume_v1"
timeframe = "4h"
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
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 2 or len(df_12h) < 55:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d candle
    # R4 = close + (high - low) * 1.5
    # R3 = close + (high - low) * 1.25
    # R2 = close + (high - low) * 1.166
    # R1 = close + (high - low) * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.083
    # S2 = close - (high - low) * 1.166
    # S3 = close - (high - low) * 1.25
    # S4 = close - (high - low) * 1.5
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.083
    r3_1d = close_1d + range_1d * 1.25
    s1_1d = close_1d - range_1d * 1.083
    s3_1d = close_1d - range_1d * 1.25
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h ADX(25) for trend strength
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_12h = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = wilder_smooth(dx, 25)
    
    # Calculate ATR(14) for 4h timeframe (for stoploss)
    tr1_4h = np.abs(high[1:] - low[1:])
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price touches S1/S3 and reverses up, in uptrend (price > EMA50, ADX > 25), with volume
            if ((close[i] >= s1_1d_aligned[i] * 0.999 and close[i] <= s1_1d_aligned[i] * 1.001) or
                (close[i] >= s3_1d_aligned[i] * 0.999 and close[i] <= s3_1d_aligned[i] * 1.001)):
                if close[i] > ema_50_12h_aligned[i] and adx_12h_aligned[i] > 25 and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
            # Short entry: price touches R1/R3 and reverses down, in downtrend (price < EMA50, ADX > 25), with volume
            elif ((close[i] >= r1_1d_aligned[i] * 0.999 and close[i] <= r1_1d_aligned[i] * 1.001) or
                  (close[i] >= r3_1d_aligned[i] * 0.999 and close[i] <= r3_1d_aligned[i] * 1.001)):
                if close[i] < ema_50_12h_aligned[i] and adx_12h_aligned[i] > 25 and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price touches R1/R3 or closes below S1
            if ((close[i] >= r1_1d_aligned[i] * 0.999 and close[i] <= r1_1d_aligned[i] * 1.001) or
                (close[i] >= r3_1d_aligned[i] * 0.999 and close[i] <= r3_1d_aligned[i] * 1.001) or
                close[i] < s1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches S1/S3 or closes above R1
            if ((close[i] >= s1_1d_aligned[i] * 0.999 and close[i] <= s1_1d_aligned[i] * 1.001) or
                (close[i] >= s3_1d_aligned[i] * 0.999 and close[i] <= s3_1d_aligned[i] * 1.001) or
                close[i] > r1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals