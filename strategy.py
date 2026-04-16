#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d KAMA trend direction with 1h volume spike and choppiness regime filter.
# Long when 1d KAMA is rising, price > 1h VWAP, volume > 2x 20-period average, and chop < 61.8 (trending regime).
# Short when 1d KAMA is falling, price < 1h VWAP, volume > 2x 20-period average, and chop < 61.8.
# Exit when 1d KAMA direction reverses or volume drops below average.
# Uses discrete position size 0.25. KAMA provides adaptive trend, volume spike confirms momentum,
# chop filter avoids whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: KAMA (adaptive trend) ===
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None  # placeholder
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i < 10:  # min_periods for ER
            er[i] = np.nan
            continue
        change_val = np.abs(close_1d[i] - close_1d[i-10])
        volatility_val = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
        er[i] = change_val / volatility_val if volatility_val != 0 else 0
    er[0] = np.nan
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA direction (1=up, -1=down, 0=flat)
    kama_dir = np.zeros_like(kama)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, np.where(kama[1:] < kama[:-1], -1, 0))
    
    # Align 1d KAMA and direction to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    # Get 1h data once before loop for VWAP and volume
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Typical price for VWAP
    typical_price_1h = (high_1h + low_1h + close_1h) / 3.0
    vp = typical_price_1h * volume_1h
    
    # Cumulative VWAP (reset periodically)
    cum_vp = np.cumsum(vp)
    cum_vol = np.cumsum(volume_1h)
    vwap = np.divide(cum_vp, cum_vol, out=np.zeros_like(cum_vp), where=cum_vol!=0)
    
    # Align 1h VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1h, vwap)
    
    # Volume moving average (20-period) on 1h
    vol_ma_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_20_1h)
    
    # Get 4h data for choppiness index
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range for choppy market calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    chop = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i < 13 or np.isnan(atr_14[i]) or hh_14[i] == ll_14[i]:
            chop[i] = 50.0  # neutral
        else:
            chop[i] = 100 * np.log10(atr_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    # Align chop to 4h timeframe (already aligned since using df_4h)
    chop_aligned = chop  # no need to align as we're using 4h data directly
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_dir_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        kama_val = kama_aligned[i]
        kama_dir_val = kama_dir_aligned[i]
        vwap_val = vwap_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        chop_val = chop_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if KAMA direction turns down or price drops below VWAP
            if kama_dir_val <= 0 or price < vwap_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if KAMA direction turns up or price rises above VWAP
            if kama_dir_val >= 0 or price > vwap_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: KAMA direction must be non-zero
            trend_filter = kama_dir_val != 0
            
            # Volume filter: volume > 2x 20-period average (1h)
            vol_filter = vol > 2.0 * vol_ma_val
            
            # Regime filter: chop < 61.8 (trending market)
            regime_filter = chop_val < 61.8
            
            # Price filter: price must be on correct side of VWAP
            price_filter_long = price > vwap_val
            price_filter_short = price < vwap_val
            
            # LONG: KAMA up, price > VWAP, volume spike, trending regime
            if (kama_dir_val > 0) and price_filter_long and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: KAMA down, price < VWAP, volume spike, trending regime
            elif (kama_dir_val < 0) and price_filter_short and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dKAMA_1hVWAP_VolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0