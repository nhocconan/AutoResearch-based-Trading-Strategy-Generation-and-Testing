#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d regime filter (ADX) and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA, indicating trend strength.
# Combined with 1d ADX trend filter (ADX > 25) and volume spikes, it filters weak signals.
# Works in both bull and bear markets by trading with the trend on higher timeframe.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d timeframe
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            smoothed = np.full_like(data, np.nan)
            if len(data) < period:
                return smoothed
            # First value is simple average
            smoothed[period-1] = np.nanmean(data[1:period])
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
            return smoothed
        
        atr = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Elder Ray on 6h timeframe: Bull Power = High - EMA13, Bear Power = Low - EMA13
    def calculate_ema(data, period):
        ema = np.full(len(data), np.nan)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema13 = calculate_ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema13[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_val > 25
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) + trend filter + volume confirmation
            if (bull_val > 0 and
                trend_filter and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 (bearish momentum) + trend filter + volume confirmation
            elif (bear_val < 0 and
                  trend_filter and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bear Power >= 0 (loss of bearish pressure) or trend weakens
            if (bear_val >= 0 or
                adx_val < 20):  # trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power <= 0 (loss of bullish pressure) or trend weakens
            if (bull_val <= 0 or
                adx_val < 20):  # trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_ADX_Volume"
timeframe = "6h"
leverage = 1.0