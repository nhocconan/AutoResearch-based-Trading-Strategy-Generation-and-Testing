#!/usr/bin/env python3
# 6h_rsi2_ema13_volume_v1
# Hypothesis: 6h RSI(2) extreme reversals with 6h EMA(13) trend filter and volume confirmation.
# RSI(2) captures short-term overextensions (bull/bear), EMA(13) filters counter-trend noise,
# volume spike confirms institutional participation. Designed for 12-37 trades/year (50-150 over 4 years).
# Works in bull/bear markets: mean reversion in ranging, trend-filtered in directional moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi2_ema13_volume_v1"
timeframe = "6h"
leverage = 1.0

def rsi(series, period):
    """Calculate RSI with Wilder's smoothing"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    delta = np.diff(series, prepend=series[0])
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    roll_down = pd.Series(down).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = np.where(roll_down != 0, roll_up / roll_down, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate 6h RSI(2)
    rsi_2 = rsi(close_6h, 2)
    # Calculate 6h EMA(13)
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 6h indicators to 6h timeframe (completed 6h candle only)
    rsi_2_aligned = align_htf_to_ltf(prices, df_6h, rsi_2)
    ema_13_aligned = align_htf_to_ltf(prices, df_6h, ema_13)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (vol_ma_20 * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_6h, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_2_aligned[i]) or np.isnan(ema_13_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI(2) > 80 (overbought) or price closes below EMA(13)
            if rsi_2_aligned[i] > 80 or close[i] < ema_13_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI(2) < 20 (oversold) or price closes above EMA(13)
            if rsi_2_aligned[i] < 20 or close[i] > ema_13_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI(2) < 10 (extreme oversold), price above EMA(13), volume spike
            if (rsi_2_aligned[i] < 10) and (close[i] > ema_13_aligned[i]) and vol_spike_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI(2) > 90 (extreme overbought), price below EMA(13), volume spike
            elif (rsi_2_aligned[i] > 90) and (close[i] < ema_13_aligned[i]) and vol_spike_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals