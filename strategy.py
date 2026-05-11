#!/usr/bin/env python3
"""
12h_WilliamsAlligator_TrixVolume_ElderRay
Hypothesis: Combine Williams Alligator (trend direction), TRIX (momentum), and Elder Ray (bull/bear power) with volume confirmation on 12h timeframe. Uses 1w trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by following the primary trend with momentum confirmation. Targets 15-30 trades/year to minimize fee drag.
"""

name = "12h_WilliamsAlligator_TrixVolume_ElderRay"
timeframe = "12h"
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
    
    # === Williams Alligator (13,8,5 SMAs) ===
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period SMMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period SMMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period SMMA
    
    # Alligator alignment: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
    bullish_alligator = (lips > teeth) & (teeth > jaw)
    bearish_alligator = (lips < teeth) & (teeth < jaw)
    
    # === TRIX (15-period) ===
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period ago
    ema1 = pd.Series(close).ewm(span=15, min_periods=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, min_periods=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, min_periods=15, adjust=False).mean()
    trix = ema3.pct_change() * 100  # percentage change
    trix_values = trix.fillna(0).values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # TRIX bullish/bearish crossover
    trix_bullish = trix_values > trix_signal
    trix_bearish = trix_values < trix_signal
    
    # === Elder Ray (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13  # High minus EMA
    bear_power = low - ema13   # Low minus EMA
    
    # Elder Ray signals
    elder_bullish = bull_power > 0
    elder_bearish = bear_power < 0
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.8  # 1.8x average volume for confirmation
    
    # === 1week EMA34 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_12h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1w trend: price above/below EMA34
    trend_up = close > ema34_1w_12h
    trend_down = close < ema34_1w_12h
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers all indicator calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(trix_values[i]) or np.isnan(trix_signal[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1w_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish Alligator + TRIX bullish cross + Elder Ray bullish + volume + 1w uptrend
            if (bullish_alligator[i] and trix_bullish[i] and elder_bullish[i] and 
                volume_ok[i] and trend_up[i]):
                signals[i] = position_size
                position = 1
            # Short: Bearish Alligator + TRIX bearish cross + Elder Ray bearish + volume + 1w downtrend
            elif (bearish_alligator[i] and trix_bearish[i] and elder_bearish[i] and 
                  volume_ok[i] and trend_down[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Opposite signal or Alligator alignment change
            if position == 1:
                # Exit long on bearish signals or Alligator turning bearish
                if (bearish_alligator[i] or trix_bearish[i] or not elder_bullish[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short on bullish signals or Alligator turning bullish
                if (bullish_alligator[i] or trix_bullish[i] or not elder_bearish[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals