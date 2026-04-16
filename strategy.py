#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 12h Donchian(20) breakout with volume confirmation
# In choppy markets (CHOP > 61.8): mean reversion at Donchian bands
# In trending markets (CHOP < 38.2): breakout continuation
# Uses daily timeframe for trend filter (price > daily EMA50 for long, < daily EMA50 for short)
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag on 12h timeframe
# Combines regime detection with price channels for robust performance in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily EMA50 filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 12h Choppiness Index (14-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (highest_high - lowest_low)) / log10(14)
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    chop_raw = np.where(hl_range > 0, atr_sum / hl_range, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop = np.where(hl_range > 0, chop, 50.0)  # Set to 50 when range is zero
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # === 12h Donchian(20) channels ===
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_1d_aligned[i]
        chop_val = chop_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume for confirmation
        
        # Regime-based logic
        if chop_val > 61.8:  # Choppy market - mean reversion
            # Long when price touches lower Donchian band AND price > daily EMA50
            if price <= donchian_low_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price touches upper Donchian band AND price < daily EMA50
            elif price >= donchian_high_val and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            # Exit mean reversion positions when price moves back toward center
            elif position == 1 and price >= (donchian_low_val + donchian_high_val) / 2:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price <= (donchian_low_val + donchian_high_val) / 2:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        elif chop_val < 38.2:  # Trending market - breakout continuation
            # Long when price breaks above Donchian high AND price > daily EMA50
            if price > donchian_high_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND price < daily EMA50
            elif price < donchian_low_val and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:  # Transition zone - no clear regime
            signals[i] = 0.0
            position = 0
    
    return signals

name = "12h_ChopRegime_Donchian20_1dEMA50_Volume1.5x"
timeframe = "12h"
leverage = 1.0