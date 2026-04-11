#!/usr/bin/env python3
"""
6h_12h_1d_williams_alligator_v1
Strategy: 6s Williams Alligator (Jaw/Teeth/Lips) with 12h/1d trend filter and volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Williams Alligator identifies trending vs ranging markets. In trending markets (JAW > TEETH > LIPS for down, JAW < TEETH < LIPS for up), trade in direction of trend with 12h/1d confirmation. In ranging markets (intertwined lines), stay flat. Volume confirmation reduces false signals. Designed to work in both bull (catch trends) and bear (avoid whipsaw via alignment) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

def williams_alligator(high, low, close):
    """Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMMA"""
    median_price = (high + low) / 2
    
    def smma(series, period):
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(series, np.nan, dtype=float)
        for i in range(len(series)):
            if i < period - 1:
                smma_vals[i] = np.nan
            elif i == period - 1:
                smma_vals[i] = sma[i]
            else:
                if np.isnan(smma_vals[i-1]) or np.isnan(sma[i]):
                    smma_vals[i] = np.nan
                else:
                    smma_vals[i] = (smma_vals[i-1] * (period - 1) + sma[i]) / period
        return smma_vals
    
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Williams Alligator on 6h
    jaw, teeth, lips = williams_alligator(high, low, close)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_avg  # Above average volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Williams Alligator conditions
        # Trending up: Lips < Teeth < Jaw (green < red < blue)
        trending_up = lips[i] < teeth[i] < jaw[i]
        # Trending down: Jaw < Teeth < Lips (blue < red < green)
        trending_down = jaw[i] < teeth[i] < lips[i]
        # Ranging: lines intertwined (not clearly separated)
        ranging = not (trending_up or trending_down)
        
        # 12h/1d trend alignment
        uptrend_12h = price_close > ema_50_12h_aligned[i]
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_12h = price_close < ema_50_12h_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = trending_up and uptrend_12h and uptrend_1d and vol_confirm[i]
        short_entry = trending_down and downtrend_12h and downtrend_1d and vol_confirm[i]
        
        # Exit when market goes ranging or trend breaks
        exit_long = position == 1 and (ranging or not uptrend_12h or not uptrend_1d)
        exit_short = position == -1 and (ranging or not downtrend_12h or not downtrend_1d)
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals