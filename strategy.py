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
    
    # Get daily data for ADX trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily ATR (14-period) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr1.iloc[0] = 0
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_ma = pd.Series(atr_14).rolling(window=10, min_periods=10).mean().values
    atr_14_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_14_ma)
    
    # Daily ADX (14-period) for trend strength
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    tr_for_adx = tr.copy()
    tr_for_adx[0] = tr_for_adx[1] if len(tr_for_adx) > 1 else 0
    
    atr_adx = pd.Series(tr_for_adx).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_adx
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_adx
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4-hour Donchian channel (20-period) for breakout signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian, volume MA, ADX, and ATR
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or \
           np.isnan(adx_aligned[i]) or np.isnan(atr_14_ma_aligned[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        atr_ma = atr_14_ma_aligned[i]
        vol_spike_val = vol_spike[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        
        # Only trade when volatility is above average (avoid chop)
        if atr_14[i] < atr_ma:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + strong trend (ADX > 25)
            if close[i] > upper and vol_spike_val and adx_val > 25:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + volume spike + strong trend (ADX > 25)
            elif close[i] < lower and vol_spike_val and adx_val > 25:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend weakens (ADX < 20)
            if close[i] < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend weakens (ADX < 20)
            if close[i] > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_ADX_Volume_v1"
timeframe = "4h"
leverage = 1.0