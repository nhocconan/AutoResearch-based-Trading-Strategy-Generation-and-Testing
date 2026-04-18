#!/usr/bin/env python3
"""
6h_Keltner_WickReversal_V1
Hypothesis: Price often reverses after piercing the Keltner Channel (ATR-based volatility band) 
with a long wick, indicating exhaustion. We combine this with a daily trend filter (EMA50/EMA200)
and volume confirmation to avoid false signals. Works in both bull and bear markets because 
wick reversals occur at exhaustion points during trends and at support/resistance in ranges.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    atr_period = 10
    kc_multiplier = 1.5
    ema_period = 20
    
    # Calculate EMA and ATR for Keltner Channel
    close_series = pd.Series(close)
    ema = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    upper_keltner = ema + kc_multiplier * atr
    lower_keltner = ema - kc_multiplier * atr
    
    # Wick rejection detection: long upper wick (sell signal) or long lower wick (buy signal)
    body_size = np.abs(close - open_) if 'open_' in locals() else np.abs(close - prices['open'].values)
    upper_wick = high - np.maximum(close, prices['open'].values)
    lower_wick = np.minimum(close, prices['open'].values) - low
    
    # Significant wick: at least 2x body size
    significant_upper_wick = upper_wick > 2 * body_size
    significant_lower_wick = lower_wick > 2 * body_size
    
    # Wick rejection conditions: price rejects Keltner band with long wick
    reject_upper = (high > upper_keltner) & significant_upper_wick  # pierced upper band but closed back inside
    reject_lower = (low < lower_keltner) & significant_lower_wick   # pierced lower band but closed back inside
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_period, atr_period, 50)  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + lower wick rejection (price bounced off lower Keltner)
            if uptrend and vol_confirm[i] and reject_lower[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + upper wick rejection (price rejected at upper Keltner)
            elif downtrend and vol_confirm[i] and reject_upper[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or upper wick rejection (exhaustion)
            if not uptrend or reject_upper[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or lower wick rejection (exhaustion)
            if not downtrend or reject_lower[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Keltner_WickReversal_V1"
timeframe = "6h"
leverage = 1.0