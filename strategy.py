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
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
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
    plus_dm_smooth = np.full(len(df_1d), np.nan)
    minus_dm_smooth = np.full(len(df_1d), np.nan)
    tr_smooth = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 14:
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
    else:
        plus_di_14 = np.full(len(df_1d), np.nan)
        minus_di_14 = np.full(len(df_1d), np.nan)
        dx_14 = np.full(len(df_1d), np.nan)
    
    adx_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 27:  # Need 14 + 14 for smoothing
        adx_14[26] = np.mean(dx_14[14:28])
        for i in range(27, len(df_1d)):
            adx_14[i] = (adx_14[i-1] * 13 + dx_14[i]) / 14
    
    # Align indicators to 1d timeframe (same as price)
    atr_1d_aligned = atr_1d  # Already at 1d frequency
    adx_1d_aligned = adx_14  # Already at 1d frequency
    
    # Calculate daily Donchian channels (20-period)
    donch_high = np.full(len(df_1d), np.nan)
    donch_low = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            donch_high[i] = np.max(high_1d[i-19:i+1])
            donch_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate daily volume moving average (20-period)
    volume_ma = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            volume_ma[i] = np.mean(volume_1d[i-19:i+1])
    
    # We need volume_1d - get it from df_1d
    volume_1d = df_1d['volume'].values
    
    # Recalculate volume_ma with actual volume_1d
    volume_ma = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            volume_ma[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily indicators to lower timeframe (price timeframe)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_aligned)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_aligned)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_aligned[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 70% of 20-period MA)
        if volume[i] < 0.7 * volume_ma_aligned[i]:
            signals[i] = 0.0
            continue
        
        # Skip low trend strength (ADX < 25)
        if adx_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume confirmation
            if close[i] > donch_high_aligned[i] and volume[i] > 1.5 * volume_ma_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low with volume confirmation
            elif close[i] < donch_low_aligned[i] and volume[i] > 1.5 * volume_ma_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily Donchian low
            if close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily Donchian high
            if close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian20_Volume_Trend_Filter"
timeframe = "1d"
leverage = 1.0