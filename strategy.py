#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter + 12h volume confirmation
# Elder Ray measures bull/bear power relative to EMA13, identifying strong directional moves
# ADX > 25 filters for trending markets where Elder Ray signals are reliable
# 12h volume spike (>1.5x 20-period average) confirms institutional participation
# Works in bull/bear: Elder Ray adapts to trend direction, volume filter reduces false signals
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_12h_elder_ray_volume_adx_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ADX and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray on 6h timeframe
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 6h
    bull_power = high - ema13  # Bull power: high minus EMA13
    bear_power = low - ema13   # Bear power: low minus EMA13
    
    # Calculate ADX on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing for TR, +DM, -DM
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr_12h, 14)
    plus_dm_12h = wilders_smoothing(plus_dm, 14)
    minus_dm_12h = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di_12h = np.where(atr_12h > 0, 100 * plus_dm_12h / atr_12h, 0)
    minus_di_12h = np.where(atr_12h > 0, 100 * minus_dm_12h / atr_12h, 0)
    
    # DX and ADX
    dx_12h = np.where((plus_di_12h + minus_di_12h) > 0, 
                      100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 
                      0)
    adx_12h = wilders_smoothing(dx_12h, 14)
    
    # Calculate 12h average volume (20-period)
    volume_12h = df_12h['volume'].values
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe (wait for 12h bar close)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5 * 20-period average volume (from 12h data)
        volume_confirmed = volume[i] > 1.5 * avg_volume_12h_aligned[i]
        
        # ADX filter: ADX > 25 indicates trending market
        trending_market = adx_12h_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: bear power becomes positive (weakening bearish pressure) OR ADX drops
            if bear_power[i] > 0 or not trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull power becomes negative (weakening bullish pressure) OR ADX drops
            if bull_power[i] < 0 or not trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: only in trending market with volume confirmation
            if trending_market and volume_confirmed:
                # Strong bullish signal: bull power > 0 and bear power < 0
                if bull_power[i] > 0 and bear_power[i] < 0:
                    position = 1
                    signals[i] = 0.25
                # Strong bearish signal: bear power < 0 and bull power < 0 (both negative)
                elif bear_power[i] < 0 and bull_power[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals