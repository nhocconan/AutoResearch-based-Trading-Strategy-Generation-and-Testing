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
    
    # === 1d Donchian channels (20-period) for breakout signals ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period high and low channels
    donch_high = np.full_like(high_1d, np.nan)
    donch_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            donch_high[i] = np.max(high_1d[i-19:i+1])
            donch_low[i] = np.min(low_1d[i-19:i+1])
        else:
            donch_high[i] = np.max(high_1d[0:i+1]) if i > 0 else high_1d[0]
            donch_low[i] = np.min(low_1d[0:i+1]) if i > 0 else low_1d[0]
    
    # === 1d ATR (14-period) for volatility filter and stoploss ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === 1d RSI (14-period) for overbought/oversold filter ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # === 1d Volume spike filter ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[0]
    
    vol_spike = volume > vol_ma_20 * 2.0  # Volume spike: 2x average
    
    # Align all indicators to daily timeframe (since we're using 1d timeframe)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume spike
        if position == 0:
            # Long: price breaks above Donchian high + RSI not overbought + volume spike
            if (close[i] > donch_high_aligned[i] and 
                rsi_14_aligned[i] < 70 and  # Not overbought
                vol_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low + RSI not oversold + volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  rsi_14_aligned[i] > 30 and  # Not oversold
                  vol_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below Donchian low OR RSI overbought
            if (close[i] < donch_low_aligned[i] or 
                rsi_14_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high OR RSI oversold
            if (close[i] > donch_high_aligned[i] or 
                rsi_14_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_RSI_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0