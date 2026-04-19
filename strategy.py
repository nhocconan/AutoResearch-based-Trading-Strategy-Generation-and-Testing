#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + chop filter - Weekly trend filter for daily signals
# Use weekly Choppiness Index to determine regime: >61.8 = range (mean revert), <38.2 = trending (trend follow).
# In trending regime: KAMA direction for trend following (long when KAMA rising, short when falling).
# In ranging regime: RSI extremes for mean reversion (long RSI<30, short RSI>70).
# Volume confirmation: volume > 1.5x 20-period average.
# Target: 15-25 trades/year per symbol to stay within frequency limits.
name = "1d_KAMA_RSI_Chop_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Wilder's smoothing function
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # ATR (14-period Wilder's smoothing)
    atr_1w = wilder_smooth(tr, 14)
    # Sum of TR over 14 periods
    tr_sum_14 = wilder_smooth(tr, 14)
    
    # Highest high and lowest low over 14 periods
    def highest_high(arr, period):
        result = np.full_like(arr, np.nan)
        for i in range(period-1, len(arr)):
            result[i] = np.max(arr[i-period+1:i+1])
        return result
    
    def lowest_low(arr, period):
        result = np.full_like(arr, np.nan)
        for i in range(period-1, len(arr)):
            result[i] = np.min(arr[i-period+1:i+1])
        return result
    
    hh_14 = highest_high(high_1w, 14)
    ll_14 = lowest_low(low_1w, 14)
    
    # Avoid division by zero
    safe_tr_sum = np.where(tr_sum_14 == 0, np.finfo(float).eps, tr_sum_14)
    chop = 100 * np.log10(safe_tr_sum / (hh_14 - ll_14)) / np.log10(14)
    # Handle cases where hh_14 == ll_14
    chop = np.where((hh_14 - ll_14) == 0, 50, chop)  # Neutral when no range
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA (Adaptive Moving Average) - ER=10, FC=2, SC=30
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        n = len(close)
        if n < er_length:
            return np.full(n, np.nan)
        
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close, n=er_length))
        abs_change = np.abs(np.diff(close))
        abs_sum = np.zeros_like(close)
        for i in range(er_length, len(close)):
            abs_sum[i] = np.sum(abs_change[i-er_length+1:i+1])
        
        er = np.zeros(n)
        er[er_length:] = change / np.where(abs_sum[er_length:] == 0, 1, abs_sum[er_length:])
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # Calculate KAMA
        kama_vals = np.full(n, np.nan)
        kama_vals[er_length] = close[er_length]
        for i in range(er_length + 1, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # RSI (14-period)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Calculate indicators
    kama_vals = kama(close_1d, er_length=10, fast_sc=2, slow_sc=30)
    rsi_vals = rsi(close_1d, period=14)
    
    # Align indicators to daily timeframe (already daily, but for consistency)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_vals)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_vals)
    
    # Get daily average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure KAMA (10+), RSI (14), chop (14*2+6), and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # KAMA slope (trend direction)
        kama_slope = kama_val - kama_aligned[i-1] if i > 0 else 0
        
        # Regime determination
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        # Neutral zone (38.2-61.8) - no trades
        
        if position == 0:
            # Determine entry based on regime
            if is_trending and volume_confirmed:
                # Trending regime: KAMA direction
                if kama_slope > 0:
                    signals[i] = 0.25
                    position = 1
                elif kama_slope < 0:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging and volume_confirmed:
                # Ranging regime: RSI extremes
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            if is_trending:
                # Exit when KAMA slope turns negative
                if kama_slope < 0:
                    exit_signal = True
            else:
                # Exit when RSI reaches overbought
                if rsi_val > 70:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            if is_trending:
                # Exit when KAMA slope turns positive
                if kama_slope > 0:
                    exit_signal = True
            else:
                # Exit when RSI reaches oversold
                if rsi_val < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals