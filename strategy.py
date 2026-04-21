#!/usr/bin/env python3
"""
1d_Weekly_HTF_Trend_Long_Only
Hypothesis: In BTC/ETH, weekly timeframe provides strong directional bias. Go long only when price is above weekly EMA34 and below 1d Donchian upper band (dip buying in uptrend). Weekly EMA acts as dynamic support in bull markets and avoids shorts in bear markets. Low trade frequency by design (~10-20/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly EMA34 trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d Donchian channel (20-period) for entry timing ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_ema = ema_34_1w_aligned[i]
        donchian_upper = donchian_high_aligned[i]
        
        if position == 0:
            # Long: price above weekly EMA (uptrend) AND touches or breaks below Donchian upper band (dip buying)
            if price_close > weekly_ema and price_close <= donchian_upper:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
        
        elif position == 1:
            # Exit: price breaks below weekly EMA (trend change) OR reaches Donchian upper band (mean reversion target)
            if price_close < weekly_ema or price_close >= donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "1d_Weekly_HTF_Trend_Long_Only"
timeframe = "1d"
leverage = 1.0