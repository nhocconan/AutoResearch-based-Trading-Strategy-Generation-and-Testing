#!/usr/bin/env python3
"""
12h_1d_RSI_Trend_Reversal_v1
Hypothesis: On 12h timeframe, take long when RSI(14) shows bullish momentum in trending markets (ADX>25) and price is above EMA(50), and short when RSI shows bearish momentum with price below EMA(50). Uses EMA for trend direction and RSI for momentum timing to capture reversals within trends. Designed for 20-40 trades/year by requiring aligned trend/momentum conditions, avoiding whipsaws in ranging markets. Works in bull markets via RSI>50 longs and bear markets via RSI<50 shorts, with EMA filter ensuring trades follow the dominant trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_Trend_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(50) for trend direction
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / loss_ma
    rs[loss_ma == 0] = np.inf  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    plus_di[tr_14 == 0] = 0
    minus_di[tr_14 == 0] = 0
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[(plus_di + minus_di) == 0] = 0
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA/RSI warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50[i]) or np.isnan(rsi[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade with the trend (ADX > 25 on 1d)
        trending = adx_aligned[i] > 25
        
        # Trend direction from EMA(50)
        above_ema = close[i] > ema50[i]
        below_ema = close[i] < ema50[i]
        
        # Momentum from RSI(14)
        rsi_bullish = rsi[i] > 50  # Bullish momentum
        rsi_bearish = rsi[i] < 50  # Bearish momentum
        
        # Entry conditions: trend + momentum alignment
        long_entry = trending and above_ema and rsi_bullish
        short_entry = trending and below_ema and rsi_bearish
        
        # Exit conditions: opposite momentum or trend weakness
        long_exit = not rsi_bullish or adx_aligned[i] < 20  # RSI turns bearish or trend weakens
        short_exit = not rsi_bearish or adx_aligned[i] < 20  # RSI turns bullish or trend weakens
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals