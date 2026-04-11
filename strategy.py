#!/usr/bin/env python3
"""
6h_12h_1d_williams_alligator_trend_v1
Strategy: 6s Williams Alligator trend following with 12h/1d trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: The Williams Alligator (Jaw, Teeth, Lips) identifies trending markets. 
Enter long when Lips > Teeth > Jaw (bullish alignment) and price above 12h EMA50.
Enter short when Lips < Teeth < Jaw (bearish alignment) and price below 12h EMA50.
Use 1d ADX > 25 to filter for trending conditions only, avoiding whipsaws in ranging markets.
Designed to capture sustained trends in both bull and bear markets while avoiding chop.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_williams_alligator_trend_v1"
timeframe = "6h"
leverage = 1.0

def williams_alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price"""
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
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
    
    # 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.concatenate([[np.nan], np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                                                  np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)])
    minus_dm = np.concatenate([[np.nan], np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                                                  np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Trend filters
        uptrend_12h = price_close > ema_50_12h_aligned[i]
        downtrend_12h = price_close < ema_50_12h_aligned[i]
        trending_market = adx_aligned[i] > 25
        
        # Entry conditions
        long_signal = bullish_alignment and uptrend_12h and trending_market
        short_signal = bearish_alignment and downtrend_12h and trending_market
        
        # Exit when Alligator alignment reverses or ADX drops below 20 (trend weakening)
        exit_long = position == 1 and (not bullish_alignment or adx_aligned[i] < 20)
        exit_short = position == -1 and (not bearish_alignment or adx_aligned[i] < 20)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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