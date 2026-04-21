# -*- coding: utf-8 -*-
# -*- mode: python; -*-
#!/usr/bin/env python3
"""
Hypothesis:
- 1d timeframe strategy combining weekly trend filter with daily price action.
- Uses weekly EMA20 for trend direction and daily Donchian breakout for entry.
- Volume confirmation and volatility filter reduce false signals.
- Designed for low trade frequency (~10-25 trades/year) to minimize fee drag.
- Works in both bull and bear markets via trend filter and breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian(20) channels for breakout signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    
    # Volume confirmation: volume / 20-period average volume
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        ema_trend = ema_20_1w_aligned[i]
        upper_band = donch_high[i]
        lower_band = donch_low[i]
        vol_ratio_val = vol_ratio[i]
        atr_ratio_val = atr_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, uptrend, volume spike, moderate volatility
            if (price_close > upper_band and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5 and 
                atr_ratio_val > 0.6 and atr_ratio_val < 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, downtrend, volume spike, moderate volatility
            elif (price_close < lower_band and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5 and 
                  atr_ratio_val > 0.6 and atr_ratio_val < 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility extremes
            if position == 1 and (price_close < lower_band or atr_ratio_val > 2.2 or atr_ratio_val < 0.4):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_band or atr_ratio_val > 2.2 or atr_ratio_val < 0.4):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA20_DonchianBreakout_VolumeATR"
timeframe = "1d"
leverage = 1.0