#!/usr/bin/env python3
# 4H_TripleConfirmation_Scalp
# Hypothesis: Combines RSI(14) mean reversion at extremes with Bollinger Band squeeze breakout
# and volume confirmation. Uses 1d ADX for regime filtering (ADX>25 = trend, <20 = range).
# Long when RSI<30 + BB squeeze breakout up + volume spike + ADX>25.
# Short when RSI>70 + BB squeeze breakout down + volume spike + ADX>25.
# Designed for 4h timeframe to capture reversal moves in trending markets with low trade frequency.
# Works in both bull and bear markets by following trend direction via ADX filter.
# Uses discrete position sizing (0.25) to minimize fee churn.

name = "4H_TripleConfirmation_Scalp"
timeframe = "4h"
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # +DM and -DM
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Bollinger Bands (20, 2) on 4h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze detection (width < 20-period average width)
    bb_width_ma = bb_width.rolling(window=20, min_periods=20).mean()
    bb_squeeze = bb_width < bb_width_ma
    
    # RSI(14) on 4h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_threshold = vol_ma * 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx_1d_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(bb_squeeze[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 for trending market
        is_trending = adx_1d_aligned[i] > 25
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper[i]
        breakout_down = close[i] < bb_lower[i]
        
        # RSI extremes
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        if position == 0:
            # Long entry: RSI oversold + BB breakout up + volume spike + trending market
            if (rsi_oversold and breakout_up and 
                volume[i] > vol_threshold[i] and is_trending):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought + BB breakout down + volume spike + trending market
            elif (rsi_overbought and breakout_down and 
                  volume[i] > vol_threshold[i] and is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 or BB breakout down or volume drops
            if (rsi_values[i] > 50 or breakout_down or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 or BB breakout up or volume drops
            if (rsi_values[i] < 50 or breakout_up or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals