#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w ADX regime filter
# - Primary timeframe: 12h for lower trade frequency (target: 50-150 total trades over 4 years)
# - Entry: Long when price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average
#          Short when price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average
# - Regime filter: Only trade in trending markets (1w ADX > 25), avoid ranging markets
# - Exit: Opposite Donchian breakout or ATR-based stoploss (signal -> 0)
# - Position size: 0.25 to limit drawdown during 2022 crash
# - Works in both bull/bear: ADX regime filter avoids whipsaws in ranging markets, Donchian captures trends

name = "12h_1d_1w_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Calculate 1w ADX(14) for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr0 = high_1w[i] - low_1w[i]
        tr1 = abs(high_1w[i] - close_1w[i-1])
        tr2 = abs(low_1w[i] - close_1w[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm = np.zeros(len(df_1w))
    minus_dm = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed DM and TR (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full(len(data), np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Calculate smoothed values
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate DI and DX
    plus_di_14 = np.full(len(df_1w), np.nan)
    minus_di_14 = np.full(len(df_1w), np.nan)
    dx_14 = np.full(len(df_1w), np.nan)
    
    for i in range(14, len(df_1w)):
        if tr_14[i] != 0:
            plus_di_14[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / tr_14[i]) * 100
            if (plus_di_14[i] + minus_di_14[i]) != 0:
                dx_14[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Align HTF data to 12h timeframe
    vol_ma_20_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_14_12h = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # Calculate Donchian channels on 12h
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(donchian_period-1, n):
        donchian_high[i] = np.max(high[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_period, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20_12h[i]) or
            np.isnan(adx_14_12h[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_20_12h[i] if vol_ma_20_12h[i] > 0 else 0
        adx = adx_14_12h[i]
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low OR ADX weakens (range)
            if close[i] <= donchian_low[i] or adx < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high OR ADX weakens (range)
            if close[i] >= donchian_high[i] or adx < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout + volume confirmation + ADX trend filter
            if adx > 25:  # Only trade in trending markets
                vol_confirm = vol_ratio > 1.5  # Volume > 1.5x average
                
                # Long entry: price breaks above Donchian high with volume confirmation
                if close[i] > donchian_high[i] and vol_confirm:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low with volume confirmation
                elif close[i] < donchian_low[i] and vol_confirm:
                    position = -1
                    signals[i] = -0.25
    
    return signals