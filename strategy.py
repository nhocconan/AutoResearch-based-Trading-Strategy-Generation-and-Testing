#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and chop regime filter
# Long when price breaks above Donchian(20) high AND close > 1d EMA34 AND volume > 2.0x 20-bar avg AND chop > 61.8 (ranging)
# Short when price breaks below Donchian(20) low AND close < 1d EMA34 AND volume > 2.0x 20-bar avg AND chop > 61.8 (ranging)
# Exit when price touches opposite Donchian(10) level or volume drops below average
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-50 trades/year on 4h.
# Donchian breakouts capture strong moves, volume confirmation ensures conviction,
# 1d EMA34 filter aligns with higher timeframe trend, chop filter avoids whipsaws in strong trends.

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian channels (10-period) for exit
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Choppiness Index: chop > 61.8 indicates ranging market (good for mean reversion/breakouts)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr_14.sum() / np.log10(range_14)) / np.log10(14) if len(atr_14) >= 14 else 50
    # Actually compute properly for each bar
    chop_values = np.full(n, 50.0)  # default to neutral
    for i in range(14, n):
        atr_sum = pd.Series(tr[i-13:i+1]).sum()
        range_val = hh_14[i] - ll_14[i]
        if range_val > 0:
            chop_values[i] = 100 * np.log10(atr_sum) / np.log10(14) / np.log10(range_val)
        else:
            chop_values[i] = 50.0
    
    chop_filter = chop_values > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10, 34, 20, 14)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or np.isnan(donchian_high_10[i]) or 
            np.isnan(donchian_low_10[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_34_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        dh_20 = donchian_high_20[i]
        dl_20 = donchian_low_20[i]
        dh_10 = donchian_high_10[i]
        dl_10 = donchian_low_10[i]
        chop_ok = chop_filter[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian(20) high AND close > 1d EMA34 AND volume confirmation AND chop > 61.8
            if curr_high > dh_20 and curr_close > ema_trend and vol_conf and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian(20) low AND close < 1d EMA34 AND volume confirmation AND chop > 61.8
            elif curr_low < dl_20 and curr_close < ema_trend and vol_conf and chop_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price touches Donchian(10) low OR volume drops below average
            if curr_low <= dl_10 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price touches Donchian(10) high OR volume drops below average
            if curr_high >= dh_10 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals