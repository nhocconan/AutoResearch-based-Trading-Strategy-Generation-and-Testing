#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v4
Hypothesis: Refine the proven Camarilla R3/S3 breakout strategy by tightening entry conditions to reduce trade frequency and avoid overtrading. Requires: price break of R3/S3, trend alignment (price vs EMA50), volume spike (>2x 20-bar average), AND momentum confirmation (RSI > 50 for longs, < 50 for shorts). This adds a momentum filter to reduce false breakouts while maintaining the core edge. Designed to work in both bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend) markets. Target: 20-40 trades per year on 4h timeframe.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v4"
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
    
    # === 1D Data for Camarilla Pivots and Trend Filter ===
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
    
    # Momentum: RSI(14) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Neutral when undefined
    
    # Align 1D indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend (price > EMA50) AND volume spike AND bullish momentum (RSI > 50)
            if close[i] > r3_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i] and rsi_values[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND downtrend (price < EMA50) AND volume spike AND bearish momentum (RSI < 50)
            elif close[i] < s3_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i] and rsi_values[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA50 OR reverses below R3
            if close[i] < ema_50_aligned[i] or close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA50 OR reverses above S3
            if close[i] > ema_50_aligned[i] or close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals