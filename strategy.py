#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla R1/S1 breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above R1 with volume spike in non-choppy market.
# Short when price breaks below S1 with volume spike in non-choppy market.
# Uses discrete position size 0.25. Camarilla levels derived from prior 1d OHLC.
# 1w timeframe for HTF context, 1d for Camarilla calculation, 12h for execution.
# Targets 50-150 total trades over 4 years to minimize fee drag.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for HTF context (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1d Indicators: Camarilla levels (R1, S1, R4, S4) ===
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # R2 = close + ((high - low) * 1.1 / 6)
    # R1 = close + ((high - low) * 1.1 / 12)
    # S1 = close - ((high - low) * 1.1 / 12)
    # S2 = close - ((high - low) * 1.1 / 6)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # === 12h Indicators: Volume and Choppiness ===
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want to trade only in trending markets (CHOP < 61.8 to avoid chop)
    # True range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM for CHOP calculation
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1  # invert to positive
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / (atr_14 + 1e-10))
    minus_di = 100 * (minus_dm_smooth / (atr_14 + 1e-10))
    
    # DX and CHOP
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    dx_ma = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(dx_ma * np.sqrt(14) / np.log(14)) / np.log10(np.sqrt(14) * 14 / np.log(14))
    
    # Align all indicators to primary timeframe (12h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))  # align as float
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 and CHOP need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        ema50 = ema50_1w_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # convert back to boolean
        chop_val = chop_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < S1 (breakdown below support) OR chop > 61.8 (choppy market)
            if (price < s1) or (chop_val > 61.8):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > R1 (breakout above resistance) OR chop > 61.8 (choppy market)
            if (price > r1) or (chop_val > 61.8):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > R1 with volume spike in trending market (CHOP < 61.8) AND above 1w EMA50 (uptrend)
            if (price > r1) and vol_spike and (chop_val < 61.8) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < S1 with volume spike in trending market (CHOP < 61.8) AND below 1w EMA50 (downtrend)
            elif (price < s1) and vol_spike and (chop_val < 61.8) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1wCamarilla_R1S1_Breakout_VolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0