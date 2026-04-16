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
    
    # === Daily OHLC for Donchian channel calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 20-period Donchian channel on daily ===
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === ATR for volatility filter (14-period) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_ma = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values  # 20-period MA of ATR
    
    # === Align HTF data to 1h timeframe ===
    donch_high_1h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_1h = align_htf_to_ltf(prices, df_1d, donch_low)
    atr_1d_ma_1h = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_1h[i]) or np.isnan(donch_low_1h[i]) or
            np.isnan(atr_1d_ma_1h[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high_1h[i]
        lower = donch_low_1h[i]
        atr_ma = atr_1d_ma_1h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when price touches opposite band or volatility drops ===
        if position == 1:  # Long position
            # Exit when price touches lower band or volatility drops significantly
            if price <= lower or atr_ma < (atr_1d_ma_1h[i-1] * 0.7 if i > 0 else atr_ma):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches upper band or volatility drops significantly
            if price >= upper or atr_ma < (atr_1d_ma_1h[i-1] * 0.7 if i > 0 else atr_ma):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band with volume spike and sufficient volatility
            if price > upper and vol_spike and atr_ma > 0:
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price breaks below lower band with volume spike and sufficient volatility
            elif price < lower and vol_spike and atr_ma > 0:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_1d_VolumeSpike_ATRFilter"
timeframe = "1h"
leverage = 1.0