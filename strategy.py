#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Weekly (1w) Donchian breakout with 1d ADX trend filter and volume confirmation.
# Uses weekly Donchian channels to capture long-term trends, filters by daily ADX > 25 to ensure
# strong trend conditions, and requires volume spikes for entry confirmation. Designed to work
# in both bull and bear markets by only taking trades in strong trending regimes, avoiding
# whipsaws in ranging markets. Targets 15-30 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donch_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Load daily data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ADX components
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly Donchian and daily ADX to 6h timeframe
    donch_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h ATR for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr1_6h[0] = 0
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_1w_aligned[i]) or 
            np.isnan(donch_low_1w_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_6h[i]
        upper = donch_high_1w_aligned[i]
        lower = donch_low_1w_aligned[i]
        adx_val = adx_1d_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter only in strong trending conditions (ADX > 25)
            if adx_val > 25:
                if price > upper and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price < lower and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position != 0:
            # Stop loss: 2 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            # Take profit: exit when price crosses opposite Donchian band
            elif position == 1 and price < lower:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > upper:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian_ADXTrend_Volume"
timeframe = "6h"
leverage = 1.0