#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ATRFilter_V1
Hypothesis: Donchian(20) breakout with volume confirmation and ATR-based trend filter on 4h timeframe.
Works in bull/bear: Breakouts capture strong moves in both directions. Volume filter ensures conviction,
ATR filter avoids whipsaws in low-volatility regimes. Uses 12h HTF for trend bias to reduce counter-trend trades.
Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for HTF trend bias
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend bias (long-term direction)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels on primary timeframe (4h)
    lookback = 20
    high_roll = prices['high'].rolling(window=lookback, min_periods=lookback).max()
    low_roll = prices['low'].rolling(window=lookback, min_periods=lookback).min()
    upper_channel = high_roll.values
    lower_channel = low_roll.values
    
    # Calculate ATR for volatility filter and stoploss
    atr_period = 14
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    volume_ok = prices['volume'].values > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Determine HTF trend bias: bullish if price > EMA34, bearish if price < EMA34
        htf_bullish = ema_34_12h_aligned[i] > 0  # EMA34 is always positive for BTC/ETH/SOL
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian channel AND HTF bullish bias AND volume confirmation
            if (price > upper_channel[i] and 
                htf_bullish and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian channel AND HTF bearish bias AND volume confirmation
            elif (price < lower_channel[i] and 
                  not htf_bullish and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA34 (trend change) or ATR-based trailing stop
            if price < ema_34_12h_aligned[i] or price < prices['high'].iloc[i-1] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA34 (trend change) or ATR-based trailing stop
            if price > ema_34_12h_aligned[i] or price > prices['low'].iloc[i-1] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0