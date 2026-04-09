#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 6h timeframe
# ADX > 25 indicates trending regime: trade in direction of Elder Ray power
# ADX <= 25 indicates ranging regime: fade extreme Elder Ray readings
# Volume confirmation ensures participation
# Designed for low trade frequency (12-37/year) to minimize fee drag
# Works in bull/bear markets: adapts to regime via ADX filter

name = "6h_1d_elder_ray_adx_volume_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = wilders_smoothing(tr, 14)
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # DI+ and DI-
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_conf = volume_confirmed[i]
        
        if position == 1:  # Long position
            if adx > 25:  # Trending regime
                # Exit long if bear power becomes stronger than bull power or ADX weakens
                if bear > bull or adx < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit long if power fades or reverses
                if bull < 0 or bear > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if adx > 25:  # Trending regime
                # Exit short if bull power becomes stronger than bear power or ADX weakens
                if bull > bear or adx < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit short if power fades or reverses
                if bear > 0 or bull < 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if adx > 25:  # Trending regime
                # Enter long if bull power > bear power and both positive
                if bull > bear and bull > 0 and vol_conf:
                    position = 1
                    signals[i] = 0.25
                # Enter short if bear power > bull power and both negative
                elif bear > bull and bear < 0 and vol_conf:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime
                # Fade extreme power readings
                if bull < -2.0 * np.std(bull_power[max(0, i-50):i]) and vol_conf:
                    position = 1
                    signals[i] = 0.25
                elif bear > 2.0 * np.std(bear_power[max(0, i-50):i]) and vol_conf:
                    position = -1
                    signals[i] = -0.25
    
    return signals