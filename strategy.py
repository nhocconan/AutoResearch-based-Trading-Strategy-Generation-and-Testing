#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
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
    
    atr_daily = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly EMA200 for trend filter (1w)
    ema200_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(df_1w)):
            ema200_1w[i] = (close_1w[i] * 2 + ema200_1w[i-1] * 198) / 200
    
    ema200_daily = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate weekly volume moving average (20-period)
    vol_ma_20_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        for i in range(19, len(df_1w)):
            vol_ma_20_1w[i] = np.mean(volume_1w[i-19:i+1])
    
    vol_ma_20_daily = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate daily Donchian channels (20-period) for entry signals
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_daily[i]) or
            np.isnan(ema200_daily[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(vol_ma_20_daily[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_daily[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20_daily[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_daily[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        # Calculate weekly pivot levels based on previous week's range
        prev_high = high_1w[i-1] if i > 0 else high_1w[0]
        prev_low = low_1w[i-1] if i > 0 else low_1w[0]
        prev_close = close_1w[i-1] if i > 0 else close_1w[0]
        prev_range = prev_high - prev_low
        
        # Weekly pivot levels (R4/S4)
        r4 = prev_close + (prev_range * 1.1 / 2)
        s4 = prev_close - (prev_range * 1.1 / 2)
        
        # Align to daily timeframe (no extra delay needed for weekly pivot)
        r4_daily = align_htf_to_ltf(prices, df_1w, np.full(len(df_1w), r4))[i]
        s4_daily = align_htf_to_ltf(prices, df_1w, np.full(len(df_1w), s4))[i]
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume confirmation and above weekly EMA200
            if close[i] > donch_high[i] and volume_ratio > vol_threshold and close[i] > ema200_daily[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low with volume confirmation and below weekly EMA200
            elif close[i] < donch_low[i] and volume_ratio > vol_threshold and close[i] < ema200_daily[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily Donchian low OR below weekly EMA200
            if close[i] < donch_low[i] or close[i] < ema200_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily Donchian high OR above weekly EMA200
            if close[i] > donch_high[i] or close[i] > ema200_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA200_R4S4_Breakout_Volume"
timeframe = "1d"
leverage = 1.0