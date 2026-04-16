#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h volume spike and chop regime filter
# Long when price > R3 AND 12h volume > 1.5x 20-period volume SMA AND chop > 61.8 (ranging market)
# Short when price < S3 AND 12h volume > 1.5x 20-period volume SMA AND chop > 61.8 (ranging market)
# Exit on opposite Camarilla touch (S3 for longs, R3 for shorts) or ATR-based stoploss
# Uses pivot levels for structure, volume confirmation for conviction, chop filter to avoid strong trends
# Discrete sizing 0.25 limits drawdown; targets 20-40 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data once before loop for volume and chop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicator: Volume SMA (20-period) for confirmation ===
    volume_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 12h Indicator: Choppiness Index (14-period) for regime filter ===
    # True Range calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(TR)/ (HH - LL)) / log10(14)
    # Avoid division by zero and log of zero/negative
    hh_minus_ll = hh_14 - ll_14
    chop_raw = np.where((hh_minus_ll > 0) & (tr_sum_14 > 0), 
                        100 * np.log10(tr_sum_14 / hh_minus_ll) / np.log10(14), 
                        50)  # neutral value when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_raw)
    
    # === 4h Indicator: Camarilla Pivot Levels (based on previous day) ===
    # Calculate daily pivot from 1d data (more accurate than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # ATR for stoploss (14-period)
    tr_4h1 = high - low
    tr_4h2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr_4h3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100  # Need 50 for chop calculation, 20 for volume SMA, 2 for 1d shift
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vol_sma_20_12h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 20-period 12h volume SMA
        vol_threshold = vol_sma_20_12h_aligned[i] * 1.5
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Chop filter: > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop_aligned[i] > 61.8
        
        # Price levels
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on Camarilla S3 touch or ATR stoploss
            if price <= s3 or price <= entry_price - 2.0 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on Camarilla R3 touch or ATR stoploss
            if price >= r3 or price >= entry_price + 2.0 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price > R3 AND volume confirmation AND chop regime
            if price > r3 and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < S3 AND volume confirmation AND chop regime
            elif price < s3 and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Camarilla_R3S3_12hVolume1.5x_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0