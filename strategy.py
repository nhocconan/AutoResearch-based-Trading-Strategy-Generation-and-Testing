#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Aroon oscillator (trend strength) with 1w ADX filter and volume confirmation.
# Aroon measures time since highest high/lowest low over N periods.
# Aroon Up > 70 and Aroon Down < 30 indicates strong uptrend (vice versa for downtrend).
# 1w ADX > 25 ensures we trade only in strong trending markets.
# Volume spike (>2x 20-period average) confirms conviction.
# Works in bull markets (Aroon Up dominant) and bear markets (Aroon Down dominant).
# Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag.
name = "1d_Aroon_1wADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Aroon calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Aroon on 1d data (25-period)
    period = 25
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # Days since highest high
    high_roll = high_1d.rolling(window=period, min_periods=period)
    idx_max = high_roll.apply(lambda x: x.argmax(), raw=True)
    aroon_up = ((period - idx_max) / period) * 100
    
    # Days since lowest low
    low_roll = low_1d.rolling(window=period, min_periods=period)
    idx_min = low_roll.apply(lambda x: x.argmin(), raw=True)
    aroon_down = ((period - idx_min) / period) * 100
    
    # Aroon oscillator: Aroon Up - Aroon Down
    aroon_osc = aroon_up.values - aroon_down.values
    
    # Align Aroon oscillator to lower timeframe (1d)
    aroon_osc_aligned = align_htf_to_ltf(prices, df_1d, aroon_osc)
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX on 1w data
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1w.diff()
    down_move = low_1w.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = dx.ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to lower timeframe (1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(aroon_osc_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Aroon conditions
        aroon_val = aroon_osc_aligned[i]
        # Strong uptrend: Aroon Up > 70 and Aroon Down < 30 -> Aroon Osc > 40
        strong_uptrend = aroon_val > 40
        # Strong downtrend: Aroon Down > 70 and Aroon Up < 30 -> Aroon Osc < -40
        strong_downtrend = aroon_val < -40
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: Strong uptrend AND strong trend AND volume spike
            if strong_uptrend and strong_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong downtrend AND strong trend AND volume spike
            elif strong_downtrend and strong_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Trend weakens (Aroon Osc < 20) OR ADX weakens
            if aroon_osc_aligned[i] < 20 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Trend weakens (Aroon Osc > -20) OR ADX weakens
            if aroon_osc_aligned[i] > -20 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals