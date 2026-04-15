#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX-based trend strength + volume-weighted RSI for mean reversion in strong trends
# Uses ADX to identify strong trending markets, then applies RSI mean-reversion within those trends.
# Volume weighting confirms institutional participation. Works in bull/bear by trading pullbacks
# in established trends rather than chasing breakouts. Target: 80-180 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ADX (14-period) on 4h for trend strength
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate RSI (14-period) on 4h
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume-weighted RSI for stronger signal
    vol_rsi = rsi * (volume_4h / (np.mean(volume_4h) + 1e-10))
    vol_rsi = np.clip(vol_rsi, 0, 100)
    
    # Align indicators to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    vol_rsi_aligned = align_htf_to_ltf(prices, df_4h, vol_rsi)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(adx_aligned[i]) or np.isnan(vol_rsi_aligned[i]):
            continue
        
        # Long: Strong trend (ADX > 25) + oversold pullback (vol_RSI < 30)
        if (adx_aligned[i] > 25 and
            vol_rsi_aligned[i] < 30 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: Strong trend (ADX > 25) + overbought pullback (vol_RSI > 70)
        elif (adx_aligned[i] > 25 and
              vol_rsi_aligned[i] > 70 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Trend weakening (ADX < 20) or RSI reverting to mean
        elif position == 1 and (adx_aligned[i] < 20 or vol_rsi_aligned[i] > 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx_aligned[i] < 20 or vol_rsi_aligned[i] < 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_ADX_VolumeRSI_MeanReversion"
timeframe = "4h"
leverage = 1.0