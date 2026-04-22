#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Keltner Channel breakout with 1d ADX trend filter and volume confirmation
    # Keltner Channel identifies volatility-based breakouts (ATR-based channels)
    # ADX > 25 filters for trending markets to avoid whipsaws in ranging conditions
    # Volume spike confirms institutional participation in the breakout
    # This combination works in both bull/bear by capturing strong trending moves
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values using Wilder's smoothing (similar to RSI)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: smoothed
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]):
                        result[i] = (result[i-1] * (period-1) + data[i]) / period
                    else:
                        result[i] = np.nan
            return result
        
        atr = wilders_smoothing(tr, period)
        dm_plus_smooth = wilders_smoothing(dm_plus, period)
        dm_minus_smooth = wilders_smoothing(dm_minus, period)
        
        # Avoid division by zero
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Keltner Channel (20-period EMA, 2*ATR)
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]):
                        result[i] = (result[i-1] * (period-1) + data[i]) / period
                    else:
                        result[i] = np.nan
            return result
        
        return wilders_smoothing(tr, period)
    
    atr_14 = calculate_atr(high, low, close, 14)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    upper_keltner = ema20 + 2 * atr_14
    lower_keltner = ema20 - 2 * atr_14
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(atr_14[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above upper Keltner + ADX > 25 (trending) + volume spike
            if close[i] > upper_keltner[i] and adx_14_1d_aligned[i] > 25 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Keltner + ADX > 25 (trending) + volume spike
            elif close[i] < lower_keltner[i] and adx_14_1d_aligned[i] > 25 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close crosses back inside Keltner Channel or ADX weakens (< 20)
            if position == 1:
                if close[i] < ema20[i] or adx_14_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema20[i] or adx_14_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Breakout_1dADX_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0