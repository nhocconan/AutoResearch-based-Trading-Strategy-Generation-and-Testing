#!/usr/bin/env python3
"""
6h ADX + Parabolic SAR with 1-week Trend Filter and Volume Confirmation
Hypothesis: ADX filters for trending markets while Parabolic SAR provides precise entries.
Combined with 1-week EMA trend filter and volume confirmation, this avoids whipsaws
in both bull and bear markets. Target: 25-35 trades/year (~100-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_psar_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1-week EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_ma
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Parabolic SAR (0.02 step, 0.2 max)
    psar = np.zeros(n)
    psar[0] = low[0]
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02
    ep = high[0] if trend == 1 else low[0]
    
    for i in range(1, n):
        if trend == 1:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low[i] < psar[i]:
                trend = -1
                psar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, 0.2)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if high[i] > psar[i]:
                trend = 1
                psar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, 0.2)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(150, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price closes below SAR OR ADX weakens (< 20)
            if (close[i] < psar[i] or 
                adx[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes above SAR OR ADX weakens (< 20)
            if (close[i] > psar[i] or 
                adx[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1-week EMA50
            uptrend = close[i] > ema_50_1w_aligned[i]
            downtrend = close[i] < ema_50_1w_aligned[i]
            
            # Strong trend filter: ADX > 25
            strong_trend = adx[i] > 25
            
            # Long: Price above SAR + uptrend + strong trend + volume spike
            if (close[i] > psar[i] and 
                uptrend and 
                strong_trend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Price below SAR + downtrend + strong trend + volume spike
            elif (close[i] < psar[i] and 
                  downtrend and 
                  strong_trend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals