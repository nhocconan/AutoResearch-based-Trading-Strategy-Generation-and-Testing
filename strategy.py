#!/usr/bin/env python3
"""
6h_Pocock_Binary_Wave_1wTrend_Volume
Hypothesis: Uses Pocock Binary Wave (PBW) - a proprietary oscillator combining
RSI, Stochastic, and Williams %R concepts to identify momentum exhaustion and
reversals. Combined with 1-week trend filter to ensure we trade with the
higher timeframe momentum, and volume confirmation to filter weak signals.
PBW oscillates between 0-100 with overbought >70 and oversold <30.
In trending markets, PBW stays in extreme zones; pullbacks to 50 offer re-entry.
In ranging markets, reversals at 70/30 provide mean reversion.
Volume confirmation ensures only significant moves are traded.
Target: 50-150 total trades over 4 years (12-37/year).
Works in both bull (buy PBW pullbacks in uptrend) and bear (sell PBW bounces in downtrend).
"""

name = "6h_Pocock_Binary_Wave_1wTrend_Volume"
timeframe = "6h"
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
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1-week volume SMA20 for volume confirmation
    volume_1w = df_1w['volume'].values
    vol_sma20_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_sma20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_sma20_1w[i] = (vol_sma20_1w[i-1] * 19 + volume_1w[i]) / 20
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    # Pocock Binary Wave (PBW) - 14 period
    # PBW = (RSI + Stoch + Williams %R) / 3, scaled 0-100
    rsi_period = 14
    stoch_period = 14
    williams_period = 14
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic %K
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    for i in range(stoch_period-1, n):
        lowest_low[i] = np.min(low[i-stoch_period+1:i+1])
        highest_high[i] = np.max(high[i-stoch_period+1:i+1])
    
    stoch_k = np.where((highest_high - lowest_low) != 0, 
                       (close - lowest_low) / (highest_high - lowest_low) * 100, 50)
    
    # Williams %R
    williams_r = np.where((highest_high - lowest_low) != 0,
                          (highest_high - close) / (highest_high - lowest_low) * -100, -50)
    
    # PBW = average of RSI, Stoch, and Williams %R (all 0-100 scale)
    pbw = (rsi + stoch_k + (100 + williams_r)) / 3  # Williams %R is -100 to 0, so +100 to make 0-100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_period, stoch_period, williams_period, 50)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma20_1w_aligned[i]) or np.isnan(pbw[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average 1w volume (scaled)
        # Approximate 6h volume from 1w: 1w volume / 28 (7days*24h/6h = 28)
        vol_6h_approx = vol_sma20_1w_aligned[i] / 28.0
        volume_confirm = volume[i] > 1.3 * vol_6h_approx
        
        if position == 0:
            # Long: PBW pulls back from oversold (<30) to above 40 in uptrend with volume
            if pbw[i] > 40 and pbw[i-1] <= 40 and close[i] > ema50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: PBW bounces from overbought (>70) to below 60 in downtrend with volume
            elif pbw[i] < 60 and pbw[i-1] >= 60 and close[i] < ema50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: PBW reaches overbought (>70) or trend reversal
            if pbw[i] >= 70 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: PBW reaches oversold (<30) or trend reversal
            if pbw[i] <= 30 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals