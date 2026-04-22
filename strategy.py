#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime. Elder Ray = Bull Power (EMA13-High), Bear Power (Low-EMA13).
# In bull regime (1d ADX>25): buy when Bull Power crosses above zero with rising trend.
# In bear regime (1d ADX>25): sell when Bear Power crosses below zero with falling trend.
# In range regime (1d ADX<20): mean revert at Bollinger Bands (2,2) on 6h.
# Uses volume confirmation (1.5x 20-period average) to filter false signals.
# Target: 20-40 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # TR = max[(H-L), abs(H-PC), abs(L-PC)]
    # +DM = H-Hprev if H-Hprev > Lprev-L and >0 else 0
    # -DM = Lprev-L if Lprev-L > H-Hprev and >0 else 0
    # SMMA = smoothed moving average
    # DX = 100 * |+DM - -DM| / (+DM + -DM)
    # ADX = SMMA of DX
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_prev)
    tr3 = np.abs(low_1d - close_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((high_1d - high_prev) > (low_prev - low_1d), np.maximum(high_1d - high_prev, 0), 0)
    dm_minus = np.where((low_prev - low_1d) > (high_1d - high_prev), np.maximum(low_prev - low_1d, 0), 0)
    
    # Smoothed moving average (Wilder's smoothing)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = WilderSmooth(tr, 14)
    dm_plus_14 = WilderSmooth(dm_plus, 14)
    dm_minus_14 = WilderSmooth(dm_minus, 14)
    
    # Avoid division by zero
    dm_sum = dm_plus_14 + dm_minus_14
    dx = np.where(dm_sum != 0, 100 * np.abs(dm_plus_14 - dm_minus_14) / dm_sum, 0)
    adx_1d = WilderSmooth(dx, 14)
    
    # Load 6h data for Elder Ray and Bollinger Bands
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # EMA13 for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_6h = high_6h - ema13_6h
    bear_power_6h = low_6h - ema13_6h
    
    # Bollinger Bands (20, 2) for mean reversion in ranging markets
    sma20_6h = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std20_6h = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    upper_bb_6h = sma20_6h + 2 * std20_6h
    lower_bb_6h = sma20_6h - 2 * std20_6h
    
    # Volume confirmation (1.5x 20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Align 6h indicators to 5m
    ema13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema13_6h)
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    upper_bb_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_bb_6h)
    lower_bb_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_bb_6h)
    sma20_6h_aligned = align_htf_to_ltf(prices, df_6h, sma20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema13_6h_aligned[i]) or 
            np.isnan(bull_power_6h_aligned[i]) or 
            np.isnan(bear_power_6h_aligned[i]) or 
            np.isnan(upper_bb_6h_aligned[i]) or 
            np.isnan(lower_bb_6h_aligned[i]) or 
            np.isnan(sma20_6h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx = adx_1d_aligned[i]
        ema13 = ema13_6h_aligned[i]
        bull_power = bull_power_6h_aligned[i]
        bear_power = bear_power_6h_aligned[i]
        upper_bb = upper_bb_6h_aligned[i]
        lower_bb = lower_bb_6h_aligned[i]
        sma20 = sma20_6h_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine regime based on 1d ADX
            if adx > 25:  # Trending market
                # Long: Bull Power crosses above zero with rising EMA13
                if i > 0 and bull_power <= 0 and bull_power_6h_aligned[i-1] < 0 and price > ema13 and vol_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power crosses below zero with falling EMA13
                elif i > 0 and bear_power >= 0 and bear_power_6h_aligned[i-1] > 0 and price < ema13 and vol_filter:
                    signals[i] = -0.25
                    position = -1
            elif adx < 20:  # Ranging market
                # Mean reversion at Bollinger Bands
                if price <= lower_bb and vol_filter:
                    signals[i] = 0.25
                    position = 1
                elif price >= upper_bb and vol_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bull Power turns negative or price reaches SMA20 (mean reversion)
                if bull_power < 0 or price >= sma20:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit when Bear Power turns positive or price reaches SMA20
                if bear_power > 0 or price <= sma20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_BollingerMeanReversion"
timeframe = "6h"
leverage = 1.0