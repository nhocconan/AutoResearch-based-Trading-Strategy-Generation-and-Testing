#!/usr/bin/env python3
"""
1h Intraday Trend + Volume + 4h Trend Filter
Hypothesis: 1h momentum with volume confirmation, filtered by 4h trend, captures intraday moves
while avoiding counter-trend trades. Volume filters out low-quality moves, 4h trend ensures
we trade with the higher timeframe momentum. Designed for low trade frequency (15-30/year).
Works in bull via trend continuation, in bear via counter-trend bounces within 4h trend context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume spike: current volume > 1.8x 20-period average (avoid too many triggers)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    # 1h price momentum: close > open (bullish candle)
    bullish_candle = close > prices['open'].values
    bearish_candle = close < prices['open'].values
    
    # Price position relative to 4h EMA: only trade in direction of 4h trend
    price_above_4h_ema = close > ema50_4h_aligned
    price_below_4h_ema = close < ema50_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        vol_ok = vol_spike[i]
        bullish = bullish_candle[i]
        bearish = bearish_candle[i]
        above_ema = price_above_4h_ema[i]
        below_ema = price_below_4h_ema[i]
        
        if position == 0:
            # Enter long: volume spike + bullish candle + price above 4h EMA (uptrend)
            if vol_ok and bullish and above_ema:
                signals[i] = 0.20
                position = 1
            # Enter short: volume spike + bearish candle + price below 4h EMA (downtrend)
            elif vol_ok and bearish and below_ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: momentum loss or price crosses below 4h EMA
            if not bullish or below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: momentum loss or price crosses above 4h EMA
            if not bearish or above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Intraday_Trend_Volume_4hFilter"
timeframe = "1h"
leverage = 1.0