#!/usr/bin/env python3
"""
4h_KAMA_RSI_BBands_Volume_v1
KAMA(10) direction + RSI(14) + Bollinger Bands(20,2) + Volume spike + ADX(14) regime filter.
Long: KAMA up, RSI>50, price above upper BB, volume>1.5x average, ADX>20.
Short: KAMA down, RSI<50, price below lower BB, volume>1.5x average, ADX>20.
Exit when KAMA reverses or RSI crosses 50.
Designed to capture momentum bursts with volatility and volume confirmation.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAMA(10) ===
    def kama(close, period=10):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / (volatility[period-1:] + 1e-10)
        sc = (er * 0.66 + 0.06) ** 2
        kama = np.zeros_like(close)
        kama[:period] = close[:period]
        for i in range(period, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10)
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50], rsi])  # pad first value
    
    # === Bollinger Bands(20,2) ===
    sma_bb = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_bb = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_bb + 2 * std_bb
    lower_bb = sma_bb - 2 * std_bb
    
    # === Volume spike (1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === ADX(14) regime filter ===
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === 1d ADX for higher timeframe trend filter ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX on daily data
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    plus_dm_d = np.where((d_high[1:] - d_high[:-1]) > (d_low[:-1] - d_low[1:]), np.maximum(d_high[1:] - d_high[:-1], 0), 0)
    minus_dm_d = np.where((d_low[:-1] - d_low[1:]) > (d_high[1:] - d_high[:-1]), np.maximum(d_low[:-1] - d_low[1:], 0), 0)
    plus_dm_d = np.concatenate([[0], plus_dm_d])
    minus_dm_d = np.concatenate([[0], minus_dm_d])
    
    tr1_d = d_high - d_low
    tr2_d = np.abs(d_high - np.roll(d_close, 1))
    tr3_d = np.abs(d_low - np.roll(d_close, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    plus_di_d = 100 * pd.Series(plus_dm_d).rolling(window=14, min_periods=14).sum().values / (atr_d * 14)
    minus_di_d = 100 * pd.Series(minus_dm_d).rolling(window=14, min_periods=14).sum().values / (atr_d * 14)
    dx_d = 100 * np.abs(plus_di_d - minus_di_d) / (plus_di_d + minus_di_d + 1e-10)
    adx_1d = pd.Series(dx_d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA up, RSI>50, price above upper BB, volume spike, ADX>20 (both timeframes)
            if (kama_vals[i] > kama_vals[i-1] and 
                rsi[i] > 50 and 
                close[i] > upper_bb[i] and 
                volume_spike[i] and 
                adx[i] > 20 and 
                adx_1d_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA down, RSI<50, price below lower BB, volume spike, ADX>20 (both timeframes)
            elif (kama_vals[i] < kama_vals[i-1] and 
                  rsi[i] < 50 and 
                  close[i] < lower_bb[i] and 
                  volume_spike[i] and 
                  adx[i] > 20 and 
                  adx_1d_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA down OR RSI < 50
            if (kama_vals[i] < kama_vals[i-1] or 
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA up OR RSI > 50
            if (kama_vals[i] > kama_vals[i-1] or 
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_BBands_Volume_v1"
timeframe = "4h"
leverage = 1.0