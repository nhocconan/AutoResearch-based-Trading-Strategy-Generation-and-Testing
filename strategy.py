# 4h_12h_TripleConfirmationBreakout_v1
# Strategy type: 4h breakout with 12h trend, volume, and volatility confirmation
# Timeframe: 4h (primary)
# Why it should work in bull AND bear: Uses multiple confirmations (trend, volume, volatility) to filter false breakouts, reducing whipsaws in ranging markets while capturing strong moves in trending markets.
# Uses 12h EMA for trend filter, volume spike for confirmation, and ATR-based volatility filter to avoid low-volatility chop.

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
    
    # Load 12h data (HTF) once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR for volatility filter (14-period)
    tr = np.zeros(len(df_12h))
    tr[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        tr[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    
    atr_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 14:
        atr_12h[13] = np.mean(tr[:14])
        for i in range(14, len(df_12h)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    atr_4h = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h EMA for trend filter (21-period)
    ema21_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 21:
        ema21_12h[20] = np.mean(close_12h[:21])
        for i in range(21, len(df_12h)):
            ema21_12h[i] = (close_12h[i] * 2 + ema21_12h[i-1] * 19) / 21
    
    ema21_4h = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 20:
        for i in range(19, len(df_12h)):
            vol_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    vol_ma_20_4h = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 4h Donchian channels (20-period) for entry signals
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
        if (np.isnan(atr_4h[i]) or
            np.isnan(ema21_4h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_4h[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20_4h[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_4h[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volume confirmation and above 12h EMA21
            if close[i] > donch_high[i] and volume_ratio > vol_threshold and close[i] > ema21_4h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low with volume confirmation and below 12h EMA21
            elif close[i] < donch_low[i] and volume_ratio > vol_threshold and close[i] < ema21_4h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low OR below 12h EMA21
            if close[i] < donch_low[i] or close[i] < ema21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high OR above 12h EMA21
            if close[i] > donch_high[i] or close[i] > ema21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_TripleConfirmationBreakout_v1"
timeframe = "4h"
leverage = 1.0