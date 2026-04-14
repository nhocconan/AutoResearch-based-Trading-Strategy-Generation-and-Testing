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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    vol_1w = df_1w['volume'].values
    
    # Calculate weekly ATR (14-period)
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly EMA (50-period)
    ema_50_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(df_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier) + (ema_50_1w[i-1] * (1 - multiplier))
    
    # Calculate weekly volatility filter (ATR > 0.8% of price)
    vol_filter_1w = np.zeros(len(df_1w))
    for i in range(len(df_1w)):
        if not np.isnan(atr_1w[i]) and close_1w[i] > 0:
            vol_filter_1w[i] = atr_1w[i] / close_1w[i] > 0.008
        else:
            vol_filter_1w[i] = False
    
    # Calculate weekly volume average (20-period)
    vol_ma_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        vol_ma_1w[19] = np.mean(vol_1w[:20])
        for i in range(20, len(df_1w)):
            vol_ma_1w[i] = (vol_ma_1w[i-1] * 19 + vol_1w[i]) / 20
    
    # Calculate volume spike filter (current volume > 1.5x 20-day average)
    vol_spike_1w = np.zeros(len(df_1w))
    for i in range(len(df_1w)):
        if not np.isnan(vol_ma_1w[i]) and vol_ma_1w[i] > 0:
            vol_spike_1w[i] = vol_1w[i] > vol_ma_1w[i] * 1.5
        else:
            vol_spike_1w[i] = False
    
    # Align indicators to daily timeframe (primary timeframe)
    atr_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_filter_1d = align_htf_to_ltf(prices, df_1w, vol_filter_1w.astype(float))
    vol_spike_1d = align_htf_to_ltf(prices, df_1w, vol_spike_1w.astype(float))
    
    # Calculate daily Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(ema_50_1d[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.8% of price)
        if vol_filter_1d[i] < 0.5:
            signals[i] = 0.0
            continue
        
        # Calculate weekly pivot levels based on previous week's range
        prev_high = high_1w[i-1] if i > 0 else high_1w[0]
        prev_low = low_1w[i-1] if i > 0 else low_1w[0]
        prev_close = close_1w[i-1] if i > 0 else close_1w[0]
        prev_range = prev_high - prev_low
        
        # Weekly pivot levels (R4/S4)
        r4 = prev_close + (prev_range * 1.1 / 2)
        s4 = prev_close - (prev_range * 1.1 / 2)
        
        # Align to daily timeframe
        r4_1d = align_htf_to_ltf(prices, df_1w, np.full(len(df_1w), r4))[i]
        s4_1d = align_htf_to_ltf(prices, df_1w, np.full(len(df_1w), s4))[i]
        
        if position == 0:
            # Long: Price breaks above daily Donchian high AND above S4 AND price > weekly EMA50 AND volume spike
            if close[i] > donch_high[i] and close[i] > s4_1d and close[i] > ema_50_1d[i] and vol_spike_1d[i] > 0.5:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low AND below R4 AND price < weekly EMA50 AND volume spike
            elif close[i] < donch_low[i] and close[i] < r4_1d and close[i] < ema_50_1d[i] and vol_spike_1d[i] > 0.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily Donchian low OR below S4 OR price < weekly EMA50
            if close[i] < donch_low[i] or close[i] < s4_1d or close[i] < ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily Donchian high OR above R4 OR price > weekly EMA50
            if close[i] > donch_high[i] or close[i] > r4_1d or close[i] > ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Camarilla_R4S4_EMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0