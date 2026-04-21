#!/usr/bin/env python3
"""
Hypothesis: 4h strategy combining Donchian channel breakout with 1d ADX trend filter and volume confirmation.
Donchian breakouts capture momentum in trending markets. ADX > 25 filters for strong trends to avoid whipsaws in ranging markets.
Volume spike (>1.5x 20-period average) confirms breakout strength. Trades with the daily trend only (long when price > daily EMA50,
short when price < daily EMA50). Uses tight entry conditions to limit trades to ~25-35 per year, reducing fee drag.
Works in bull markets via long breakouts and bear markets via short breakouts, both filtered by daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channel (20-period high/low) on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5
        
        if position == 0:
            # Enter long: price breaks above Donchian high + price > daily EMA50 (uptrend) + volume spike
            if (price_close > donchian_high[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + price < daily EMA50 (downtrend) + volume spike
            elif (price_close < donchian_low[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through Donchian channel or trend reversal
            if position == 1 and price_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0