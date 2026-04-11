#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-week Donchian channel breakout, volume confirmation, and weekly ADX trend filter.
# Designed for low trade frequency (~15-30/year) by requiring confluence of trend, breakout, and volume.
# Long when price breaks above weekly Donchian(20) high with volume > 1.5x average and weekly ADX > 25.
# Short when price breaks below weekly Donchian(20) low with volume > 1.5x average and weekly ADX > 25.
# Exit when price crosses weekly Donchian midline or ADX drops below 20.
# Works in bull/bear markets by following established trends with volatility expansion confirmation.

name = "12h_1w_donchian_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Weekly arrays
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    donch_high_20 = np.full(len(high_1w), np.nan)
    donch_low_20 = np.full(len(low_1w), np.nan)
    for i in range(19, len(high_1w)):
        donch_high_20[i] = np.max(high_1w[i-19:i+1])
        donch_low_20[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate weekly midline
    donch_mid = (donch_high_20 + donch_low_20) / 2.0
    
    # Calculate weekly ADX (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder smoothing)
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.sum(x[:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr_14w = wilders_smoothing(tr, 14)
    plus_dm_14w = wilders_smoothing(plus_dm, 14)
    minus_dm_14w = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di_14w = np.where(atr_14w != 0, 100 * plus_dm_14w / atr_14w, 0)
    minus_di_14w = np.where(atr_14w != 0, 100 * minus_dm_14w / atr_14w, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14w + minus_di_14w) != 0, 
                  100 * np.abs(plus_di_14w - minus_di_14w) / (plus_di_14w + minus_di_14w), 0)
    adx_14w = wilders_smoothing(dx, 14)
    
    # Calculate weekly average volume (20-period)
    vol_avg_20w = np.full(len(volume_1w), np.nan)
    for i in range(19, len(volume_1w)):
        vol_avg_20w[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align weekly indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_14w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * weekly average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Trend filter: weekly ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        # Exit conditions
        exit_long = (low[i] <= donch_mid_aligned[i]) or (adx_aligned[i] < 20)
        exit_short = (high[i] >= donch_mid_aligned[i]) or (adx_aligned[i] < 20)
        
        # Entry conditions
        long_break = high[i] > donch_high_aligned[i] and vol_filter and trend_filter
        short_break = low[i] < donch_low_aligned[i] and vol_filter and trend_filter
        
        if long_break and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_break and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals