#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter and volume spike confirmation. 
In bull/bear markets, price tends to continue in the direction of the weekly trend after breaking key daily pivot levels with above-average volume. 
Uses discrete position sizing (0.25) to minimize fee churn and targets 30-100 trades over 4 years.
Weekly trend filter avoids counter-trend trades during strong moves, while volume spike confirms institutional participation.
Works in ranging markets via ADX filter: only trade when ADX > 20 to avoid false breakouts in low volatility.
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ATR for volatility filter
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate ADX for regime filter (avoid ranging markets)
    adx_period = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    tr_14 = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values / tr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values / tr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    weekly_trend = np.where(close > ema_20_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_multiplier = 1.1 / 12
    rng = high - low
    R1 = close + rng * camarilla_multiplier
    S1 = close - rng * camarilla_multiplier
    
    # Volume spike filter: volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for EMA/volume MA, 14 for ADX/ATR)
    start_idx = max(20, adx_period, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(weekly_trend[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Only trade when ADX > 20 (avoid low volatility ranging markets)
        if adx[i] <= 20:
            # Exit position in low volatility
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above R1 with volume spike and weekly uptrend
        if close[i] > R1[i] and volume_spike[i] and weekly_trend[i] == 1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        
        # Short signal: price breaks below S1 with volume spike and weekly downtrend
        elif close[i] < S1[i] and volume_spike[i] and weekly_trend[i] == -1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        
        # Exit conditions: price returns to mid-point or opposite Camarilla level
        elif position == 1 and (close[i] < (R1[i] + S1[i]) / 2 or close[i] < S1[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > (R1[i] + S1[i]) / 2 or close[i] > R1[i]):
            signals[i] = 0.0
            position = 0
        
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0