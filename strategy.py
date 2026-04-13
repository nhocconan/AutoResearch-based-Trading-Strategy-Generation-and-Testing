#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. In trending markets (ADX>25),
# strong bull/bear power indicates trend continuation. In ranging markets (ADX<20),
# we fade extremes. Volume confirmation filters low-conviction moves.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for regime filter (ADX) and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth(values, period):
            smoothed = np.full_like(values, np.nan)
            if len(values) < period:
                return smoothed
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
            return smoothed
        
        atr = smooth(tr, period)
        plus_dm_smooth = smooth(plus_dm, period)
        minus_dm_smooth = smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # EMA13 on 1d for Elder Ray
    ema13_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (13 + 1)
    ema13_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema13_1d[i] = (close_1d[i] - ema13_1d[i-1]) * ema_multiplier + ema13_1d[i-1]
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Elder Ray on 6h timeframe: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # We need EMA13 on 6h for Elder Ray calculation
    ema13_6h = np.zeros(n)
    ema13_6h[0] = close[0]
    for i in range(1, n):
        ema13_6h[i] = (close[i] - ema13_6h[i-1]) * ema_multiplier + ema13_6h[i-1]
    
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        ema13_val = ema13_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Determine regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_val > 25:  # Trending market
                # Long: Strong bull power + above EMA13 + volume confirmation
                if (bull_val > 0 and close[i] > ema13_val and volume_confirm):
                    position = 1
                    signals[i] = position_size
                # Short: Strong bear power + below EMA13 + volume confirmation
                elif (bear_val < 0 and close[i] < ema13_val and volume_confirm):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:  # Ranging market (ADX < 25, use 20 as threshold for entry)
                # Fade extremes: Long when bear power is very negative (oversold)
                # Short when bull power is very high (overbought)
                if (bear_val < -np.std(bear_power[max(0, i-50):i]) * 1.5 and 
                    close[i] < ema13_val and volume_confirm):
                    position = 1
                    signals[i] = position_size
                elif (bull_val > np.std(bull_power[max(0, i-50):i]) * 1.5 and 
                      close[i] > ema13_val and volume_confirm):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Exit long: Power fades or reverses
            if (bull_val < 0 or close[i] < ema13_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Power fades or reverses
            if (bear_val > 0 or close[i] > ema13_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_ADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0