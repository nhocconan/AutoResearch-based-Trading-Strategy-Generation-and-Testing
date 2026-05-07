#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_Regime
# Hypothesis: Uses Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d trend filter.
# Long when Bull Power > 0 and Bear Power < 0 (bullish) and 1d EMA50 > 1d EMA200 (uptrend).
# Short when Bull Power < 0 and Bear Power > 0 (bearish) and 1d EMA50 < 1d EMA200 (downtrend).
# Includes volume confirmation and ATR-based volatility filter to avoid whipsaws.
# Designed for low trade frequency (~15-25/year) to work in both bull and bear markets via trend alignment.
# Elder Ray captures institutional buying/selling pressure; 1d EMA crossover filters counter-trend trades.
# Works in bull markets via long signals, in bear markets via short signals when trend confirms.
timeframe = "6h"
name = "6h_ElderRay_BullBearPower_Regime"
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
    
    # Elder Ray parameters
    ema_period = 13
    
    # Calculate EMA for Elder Ray
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema
    bear_power = ema - low
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(ema_period, 200), n):
        # Skip if any critical value is NaN
        if (np.isnan(ema[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility periods
        if atr[i] < 0.5 * np.mean(atr[max(0, i-20):i+1]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, volume spike, and 1d uptrend
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                volume[i] > 1.5 * vol_ma[i] and 
                ema_50_1d_aligned[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, volume spike, and 1d downtrend
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  ema_50_1d_aligned[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bear Power becomes positive (bullish momentum fading) or 1d trend breaks
            if bear_power[i] >= 0 or ema_50_1d_aligned[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bull Power becomes negative (bearish momentum fading) or 1d trend breaks
            if bull_power[i] <= 0 or ema_50_1d_aligned[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals