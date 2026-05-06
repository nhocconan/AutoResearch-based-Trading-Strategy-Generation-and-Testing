#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with 4h trend filter and volume confirmation
# Uses Kaufman Adaptive Moving Average (KAMA) on 1d timeframe for trend direction
# Requires price to be above/below KAMA with efficiency ratio (ER) threshold
# Uses 4h ADX(25) to filter for trending markets only
# Volume confirmation (>1.5x 20-bar average) ensures participation
# KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in both bull/bear: adapts to volatility, avoids false signals in consolidation

name = "1d_KAMA_4hADX25_VolumeConfirm_v1"
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
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1d) < 30 or len(df_4h) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate KAMA on 1d timeframe
    def kama(data, er_period=10, fast_sc=2, slow_sc=30):
        # Calculate Efficiency Ratio (ER)
        change = np.abs(np.diff(data, n=er_period))
        volatility = np.sum(np.abs(np.diff(data)), axis=0) if len(data) > 1 else 0
        # For rolling calculation, we need to compute volatility over er_period
        er = np.full_like(data, np.nan)
        for i in range(er_period, len(data)):
            price_change = np.abs(data[i] - data[i-er_period])
            price_volatility = np.sum(np.abs(np.diff(data[i-er_period:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 0
        
        # Smoothing constants
        fast_sc = 2.0 / (fast_sc + 1)
        slow_sc = 2.0 / (slow_sc + 1)
        sc = np.where(er > 0, er * (fast_sc - slow_sc) + slow_sc, 0)
        
        # Calculate KAMA
        kama_val = np.full_like(data, np.nan)
        kama_val[er_period] = data[er_period]  # Start with first valid ER point
        for i in range(er_period + 1, len(data)):
            if not np.isnan(sc[i]) and not np.isnan(kama_val[i-1]):
                kama_val[i] = kama_val[i-1] + sc[i] * (data[i] - kama_val[i-1])
            else:
                kama_val[i] = kama_val[i-1]
        return kama_val
    
    kama_1d = kama(close_1d, er_period=10, fast_sc=2, slow_sc=30)
    
    # Calculate 4h ADX(25) trend filter
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_4h = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    # DI+ = 100 * smoothed +DM / ATR, DI- = 100 * smoothed -DM / ATR
    di_plus = np.where(atr_4h != 0, 100 * dm_plus_smooth / atr_4h, 0)
    di_minus = np.where(atr_4h != 0, 100 * dm_minus_smooth / atr_4h, 0)
    
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX = smoothed DX
    adx_4h = wilder_smooth(dx, 25)
    
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
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(atr_1d[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above KAMA AND trending market (ADX > 25) AND volume confirmation
            if (close[i] > kama_1d_aligned[i] and adx_4h_aligned[i] > 25 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA AND trending market (ADX > 25) AND volume confirmation
            elif (close[i] < kama_1d_aligned[i] and adx_4h_aligned[i] > 25 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below KAMA
            if close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above KAMA
            if close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals