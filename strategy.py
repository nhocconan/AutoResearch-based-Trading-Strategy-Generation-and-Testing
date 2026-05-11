#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Refined version focusing on fewer, higher-quality trades by tightening volume confirmation and adding a momentum filter. Uses 1d RSI to avoid overextended breakouts. Target: 20-40 trades per year on 4h timeframe.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D Data for Camarilla Pivots, Trend Filter, and Momentum ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels: R3/S3 = C ± (H-L) * 1.1/2
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 2
    s3 = prev_close - rang * 1.1 / 2
    
    # Trend filter: EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Momentum filter: RSI(14) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1D indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume spike: current volume > 2.5x 20-period average (tighter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend (price > EMA50) AND volume spike AND not overbought (RSI < 70)
            if close[i] > r3_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i] and rsi_aligned[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND downtrend (price < EMA50) AND volume spike AND not oversold (RSI > 30)
            elif close[i] < s3_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i] and rsi_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA50 OR reverses below R3 OR RSI > 70 (overbought)
            if close[i] < ema_50_aligned[i] or close[i] < r3_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA50 OR reverses above S3 OR RSI < 30 (oversold)
            if close[i] > ema_50_aligned[i] or close[i] > s3_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals