#!/usr/bin/env python3
# 1h_4d_1d_Trend_Momentum_Breakout
# Hypothesis: 1h breakouts aligned with 4h trend and 1d momentum capture directional moves in both bull and bear markets.
# Uses 4h EMA for trend filter, 1d RSI for momentum filter, and 1h Donchian breakout for entry.
# Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag.
# Works in bull/bear via 4h trend filter - only trade with the 4h trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_Trend_Momentum_Breakout"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 4h: EMA21 for trend filter ===
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d: RSI14 for momentum filter ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1h: Donchian channel (20-period) for breakout ===
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        ema_4h_val = ema_4h_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with 4h uptrend and 1d bullish momentum
            if (close_val > donch_high_val and 
                close_val > ema_4h_val and  # Price above 4h EMA (uptrend)
                rsi_1d_val > 50):  # 1d RSI bullish (>50)
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Donchian low with 4h downtrend and 1d bearish momentum
            elif (close_val < donch_low_val and 
                  close_val < ema_4h_val and  # Price below 4h EMA (downtrend)
                  rsi_1d_val < 50):  # 1d RSI bearish (<50)
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or loses 4h uptrend
            if close_val < donch_low_val or close_val < ema_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or loses 4h downtrend
            if close_val > donch_high_val or close_val > ema_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals