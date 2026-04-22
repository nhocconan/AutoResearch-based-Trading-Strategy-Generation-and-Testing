#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with volume confirmation and 1w ADX trend filter.
# Long when price breaks above weekly Donchian high (20) with volume > 1.5x 20-day average and weekly ADX > 25.
# Short when price breaks below weekly Donchian low (20) with volume > 1.5x 20-day average and weekly ADX > 25.
# Weekly ADX ensures we only trade in trending markets, avoiding choppy conditions.
# Weekly Donchian provides structural breakout signals aligned with higher timeframe trend.
# Designed to work in both bull and bear markets by filtering with weekly ADX.
# Targets 10-25 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Donchian and ADX (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on weekly data
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period ADX on weekly data
    # ADX requires +DI, -DI, and DX calculations
    # +DM = max(0, high_t - high_{t-1}) if high_t - high_{t-1} > low_{t-1} - low_t else 0
    # -DM = max(0, low_{t-1} - low_t) if low_{t-1} - low_t > high_t - high_{t-1} else 0
    # TR = max(high-low, |high-close_{t-1}|, |low-close_{t-1}|)
    # +DI = 100 * EMA(+DM) / ATR
    # -DI = 100 * EMA(-DM) / ATR
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    # ADX = EMA(DX)
    
    # Calculate directional movement
    high_diff = np.diff(high_1w, prepend=high_1w[0])
    low_diff = -np.diff(low_1w, prepend=low_1w[0])  # negative of low diff for positive values
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate True Range
    high_low = high_1w - low_1w
    high_close_prev = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    low_close_prev = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    
    # Calculate ATR (14-period EMA of TR)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)  # avoid division by zero
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 20-day average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trending = adx_val > 25
        
        if position == 0:
            # Long conditions: price breaks above weekly Donchian high + volume spike + trending
            if price > donchian_high and vol_spike and trending:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below weekly Donchian low + volume spike + trending
            elif price < donchian_low and vol_spike and trending:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below weekly Donchian low or ADX drops below 20
                if price < donchian_low or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above weekly Donchian high or ADX drops below 20
                if price > donchian_high or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0