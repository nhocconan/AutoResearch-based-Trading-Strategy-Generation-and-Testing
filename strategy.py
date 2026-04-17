#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian Channel (20 periods) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # === 12h Volume Spike (2.0x 20-period average) ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # === 1d RSI (14 periods) for overbought/oversold filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for calculations
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-period average
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm = vol_12h_aligned[i] > vol_ma_20_12h_aligned[i] * 2.0
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_ok = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume confirmation and RSI not overbought
            if close[i] > donchian_high_aligned[i] and vol_confirm and rsi_ok:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 4h Donchian low with volume confirmation and RSI not oversold
            elif close[i] < donchian_low_aligned[i] and vol_confirm and rsi_ok:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to opposite Donchian level
        elif position == 1:
            # Exit long: price crosses below 4h Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 4h Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hVolumeSpike_RSIFilter"
timeframe = "4h"
leverage = 1.0