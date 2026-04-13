#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot levels from 1d + volume confirmation + ADX regime filter
    # Long: price > H3 (Camarilla resistance) AND volume > 1.5x avg AND ADX > 25 (trending)
    # Short: price < L3 (Camarilla support) AND volume > 1.5x avg AND ADX > 25 (trending)
    # Exit: price crosses back to H4/L4 (pivot extremes) OR ADX < 20 (range) OR volume dry-up
    # Using 4h timeframe for optimal trade frequency, Camarilla for precise pivot levels,
    # ADX for trend regime filter (avoid whipsaws in ranging markets), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily OHLC
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = Pivot + Range * 1.1/2
    # H3 = Pivot + Range * 1.1/4
    # L3 = Pivot - Range * 1.1/4
    # L4 = Pivot - Range * 1.1/2
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    h4_1d = pivot_1d + range_1d * 1.1 / 2.0
    h3_1d = pivot_1d + range_1d * 1.1 / 4.0
    l3_1d = pivot_1d - range_1d * 1.1 / 4.0
    l4_1d = pivot_1d - range_1d * 1.1 / 2.0
    
    # Align daily Camarilla levels to 4h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Calculate ADX(14) on 4h for regime filter
    # ADX = smoothed DX, where DX = |+DI - -DI| / (+DI + -DI) * 100
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = smoothed +DM / ATR * 100
    # -DI = smoothed -DM / ATR * 100
    
    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]
    
    up_move = high - prev_high
    down_move = prev_low - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di_smoothed = wilders_smoothing(plus_dm, period)
    minus_di_smoothed = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    dx = np.where((plus_di_smoothed + minus_di_smoothed) > 0,
                  np.abs(plus_di_smoothed - minus_di_smoothed) / (plus_di_smoothed + minus_di_smoothed) * 100,
                  0)
    adx = wilders_smoothing(dx, period)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market
        trending = adx[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla level break + trend + volume
        long_entry = (close[i] > h3_1d_aligned[i]) and trending and vol_confirm
        short_entry = (close[i] < l3_1d_aligned[i]) and trending and vol_confirm
        
        # Exit logic: price crosses H4/L4 OR ADX < 20 (range) OR volume dry-up
        long_exit = (close[i] >= h4_1d_aligned[i]) or (adx[i] < 20) or not vol_confirm
        short_exit = (close[i] <= l4_1d_aligned[i]) or (adx[i] < 20) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_adx_volume_v1"
timeframe = "4h"
leverage = 1.0