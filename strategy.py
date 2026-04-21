#!/usr/bin/env python3
"""
Hypothesis: 4h price action within 12h ATR-based Donchian channel with volume confirmation.
In low volatility regimes (ATR contraction), price tends to revert to the mean of the
12h ATR-based channel. When volatility expands (ATR expansion) and price breaks
the channel with volume confirmation, it signals a momentum move. Uses 12h EMA50
for trend filter to avoid counter-trend trades. Designed for ~20-40 trades/year
to minimize fee drag, works in bull/bear via trend filter and volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend and volatility
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h ATR(14) for volatility-based channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channel: 20-period high/low of ATR-adjusted price
    # We use price +/- ATR to create dynamic channels
    upper_channel = close_12h + 2 * atr_14
    lower_channel = close_12h - 2 * atr_14
    upper_channel = pd.Series(upper_channel).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(lower_channel).rolling(window=20, min_periods=20).min().values
    
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Volume confirmation: volume / 20-period average volume (12h)
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_12h_aligned[i]
        upper_ch = upper_channel_aligned[i]
        lower_ch = lower_channel_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: price breaks above upper channel, volume spike, uptrend
            if (price_close > upper_ch and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel, volume spike, downtrend
            elif (price_close < lower_ch and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite channel or trend reversal
            if position == 1 and (price_close < lower_ch or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_ch or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ATR_Donchian_Channel_Volume_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0