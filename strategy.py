#!/usr/bin/env python3
"""
4h ATR Breakout with Volume and Daily Trend Filter
Hypothesis: Price breakouts beyond ATR-based channels with volume confirmation
and alignment to daily trend capture strong momentum moves while avoiding
counter-trend trades. This strategy targets 20-30 trades/year to minimize
fee drag in both bull and bear markets by using volatility-based entry
and trend filtering.
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
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR for volatility-based channels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate upper and lower channels (ATR multiples from close)
    upper_channel = close + (1.5 * atr)
    lower_channel = close - (1.5 * atr)
    
    # Volume filter: current volume > 1.8x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        trend = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume, in uptrend
            if price > upper_channel[i] and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume, in downtrend
            elif price < lower_channel[i] and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to middle or trend weakens
            if price < close[i] or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to middle or trend weakens
            if price > close[i] or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ATR_Breakout_Volume_DailyTrend"
timeframe = "4h"
leverage = 1.0