#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d regime filter and 1w volume confirmation.
# Elder Ray = Bull Power (High - EMA13) and Bear Power (EMA13 - Low).
# In bull regime (price > 1d EMA50): go long when Bull Power rises and volume confirms.
# In bear regime (price < 1d EMA50): go short when Bear Power rises and volume confirms.
# Uses 1w volume spike to filter for institutional participation.
# Works in both bull and bear by adapting to regime.
# Target: 20-40 trades/year by requiring regime alignment + Elder Ray expansion + volume spike.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d for EMA50 (regime) and EMA13 (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA50 for regime filter
    close_d = df_1d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily EMA13 for Elder Ray
    ema13_d = pd.Series(close_d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Weekly volume for confirmation
    vol_w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_w).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_d)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_d)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate Elder Ray components using 6h high/low and aligned daily EMA13
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema13_1d_aligned  # High - EMA13
    bear_power = ema13_1d_aligned - low   # EMA13 - Low
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        vol_current = align_htf_to_ltf(prices, df_1w, vol_w)[i]  # weekly volume aligned to 6h
        
        # Regime filter: price vs daily EMA50
        bull_regime = prices['close'].iloc[i] > ema50_1d_aligned[i]
        bear_regime = prices['close'].iloc[i] < ema50_1d_aligned[i]
        
        # Elder Ray strength: current vs previous bar
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_rising = bear_power[i] > bear_power[i-1]
        
        # Volume confirmation: current volume > 1.8x 20-week average
        volume_confirm = vol_current > 1.8 * vol_ma_20_1w_aligned[i]
        
        if position == 0:
            # Enter long in bull regime when Bull Power rising with volume
            if bull_regime and bull_rising and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short in bear regime when Bear Power rising with volume
            elif bear_regime and bear_rising and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bear Power starts rising (shift to bearish) OR volume drops
                if bear_power[i] > bear_power[i-1]:  # Bear Power rising
                    exit_signal = True
                elif vol_current < vol_ma_20_1w_aligned[i]:  # Volume drop
                    exit_signal = True
            elif position == -1:
                # Exit short: Bull Power starts rising (shift to bullish) OR volume drops
                if bull_power[i] > bull_power[i-1]:  # Bull Power rising
                    exit_signal = True
                elif vol_current < vol_ma_20_1w_aligned[i]:  # Volume drop
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dRegime_1wVolume"
timeframe = "6h"
leverage = 1.0