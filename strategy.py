#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Camarilla R3/S3 levels act as key reversal points in ranging markets. 
Breakouts beyond R3/S3 with volume spike indicate strong momentum. 
Daily EMA34 filter ensures trades align with higher timeframe trend. 
Dynamic position sizing based on volatility (ATR) to manage risk. 
Works in both bull and bear markets by capturing breakouts from key levels.
Target: 20-40 trades/year per symbol.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate ATR for volatility-based sizing and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Daily EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = df_1d['close'].values > ema_34_1d
    downtrend_1d = df_1d['close'].values < ema_34_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Calculate Camarilla levels for each 4h bar using prior day's OHLC
    # We need to map each 4h bar to the prior day's OHLC
    # First, get daily OHLC
    df_1d_ohlc = get_htf_data(prices, '1d')[['open', 'high', 'low', 'close']]
    # Align to 4h: each 4h bar gets the prior day's OHLC
    prior_day_open = align_htf_to_ltf(prices, df_1d_ohlc, df_1d_ohlc['open'].values)
    prior_day_high = align_htf_to_ltf(prices, df_1d_ohlc, df_1d_ohlc['high'].values)
    prior_day_low = align_htf_to_ltf(prices, df_1d_ohlc, df_1d_ohlc['low'].values)
    prior_day_close = align_htf_to_ltf(prices, df_1d_ohlc, df_1d_ohlc['close'].values)
    
    # Calculate Camarilla levels
    # R4 = Close + (High - Low) * 1.5
    # R3 = Close + (High - Low) * 1.25
    # S3 = Close - (High - Low) * 1.25
    # S4 = Close - (High - Low) * 1.5
    rng = prior_day_high - prior_day_low
    r3 = prior_day_close + rng * 1.25
    s3 = prior_day_close - rng * 1.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Dynamic position size based on volatility (inverse ATR)
        # Base size 0.25, scaled by ATR relative to median
        if i >= 50:
            atr_median = np.median(atr[50:i+1]) if np.any(~np.isnan(atr[50:i+1])) else atr[i]
            if atr_median > 0:
                vol_factor = np.clip(atr[50] / atr_median, 0.5, 2.0)  # Normalize to first valid ATR
            else:
                vol_factor = 1.0
            base_size = 0.25
            size = base_size * vol_factor
            size = np.clip(size, 0.15, 0.35)  # Keep within reasonable bounds
        else:
            size = 0.25
        
        if position == 0:
            # LONG: Close > R3, volume spike, 1d uptrend
            if close[i] > r3[i] and volume_spike[i] and uptrend_1d_aligned[i]:
                signals[i] = size
                position = 1
            # SHORT: Close < S3, volume spike, 1d downtrend
            elif close[i] < s3[i] and volume_spike[i] and downtrend_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < S3 or loss of 1d uptrend
            if close[i] < s3[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # EXIT SHORT: Close > R3 or loss of 1d downtrend
            if close[i] > r3[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals