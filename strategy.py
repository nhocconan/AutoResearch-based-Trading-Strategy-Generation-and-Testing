# 1. Hypothesis: 4h Donchian breakout + volume confirmation + ADX trend filter
# - Entry on breakouts of 20-period Donchian channels (trend following)
# - Volume filter: requires volume > 20-period average to avoid false breakouts
# - ADX filter: only trade when ADX > 25 (strong trend)
# - Position size: 0.25 (25%) to manage drawdown in volatile markets
# - Exit when price re-enters the Donchian channel or ADX weakens (< 20)
# - Works in bull/bear by capturing trends in both directions with strict filters to avoid overtrading
# - Target: 20-50 trades/year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) - Wilder's smoothing
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily ADX (14-period) - Wilder's smoothing
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = tr
    plus_di_14 = np.full(len(df_1d), np.nan)
    minus_di_14 = np.full(len(df_1d), np.nan)
    dx_14 = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 14:
        # Smooth +DM, -DM, TR
        plus_dm_smooth = np.full(len(df_1d), np.nan)
        minus_dm_smooth = np.full(len(df_1d), np.nan)
        tr_smooth = np.full(len(df_1d), np.nan)
        
        plus_dm_smooth[13] = np.sum(plus_dm[1:15])
        minus_dm_smooth[13] = np.sum(minus_dm[1:15])
        tr_smooth[13] = np.sum(tr[1:15])
        
        for i in range(14, len(df_1d)):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
        
        plus_di_14 = 100 * plus_dm_smooth / tr_smooth
        minus_di_14 = 100 * minus_dm_smooth / tr_smooth
        dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    
    adx_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 27:  # Need 14 + 14 for smoothing
        adx_14[26] = np.mean(dx_14[14:28])
        for i in range(27, len(df_1d)):
            adx_14[i] = (adx_14[i-1] * 13 + dx_14[i]) / 14
    
    # Align indicators to 4h timeframe
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 4-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 4-hour volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_4h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(adx_4h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_4h[i] / close[i] < 0.003:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 70% of 20-period MA)
        if volume[i] < 0.7 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Skip low trend strength (ADX < 25)
        if adx_4h[i] < 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high
            if close[i] > donch_high[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low
            elif close[i] < donch_low[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low OR ADX < 20
            if close[i] < donch_low[i] or adx_4h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high OR ADX < 20
            if close[i] > donch_high[i] or adx_4h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian20_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0