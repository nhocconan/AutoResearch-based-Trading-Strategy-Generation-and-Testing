#!/usr/bin/env python3
"""
12h_1d_keltner_breakout_volume
Strategy: 12h price action with 1d Keltner Channel confluence
Timeframe: 12h
Leverage: 1.0
Hypothesis: Buy when 12h closes above upper Keltner Channel with volume confirmation and price above 200 EMA; sell when 12h closes below lower Keltner Channel with volume confirmation and price below 200 EMA. Uses 1d EMA200 as trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend. Low-frequency design targets 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_breakout_volume"
timeframe = "12h"
leverage = 1.0

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
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 12h ATR for Keltner Channel (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA for center line (20-period)
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 12h Keltner Channels
    keltner_upper = ema_20 + (2.0 * atr_12h)
    keltner_lower = ema_20 - (2.0 * atr_12h)
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA200 (trend filter) ===
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 12h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price vs 1d EMA200
        price_above_ema200 = price_close > ema_200_1d_aligned[i]
        price_below_ema200 = price_close < ema_200_1d_aligned[i]
        
        # Long conditions: 12h closes above upper Keltner with volume + price above 1d EMA200
        long_signal = volume_confirmed and (price_close > keltner_upper[i]) and price_above_ema200
        
        # Short conditions: 12h closes below lower Keltner with volume + price below 1d EMA200
        short_signal = volume_confirmed and (price_close < keltner_lower[i]) and price_below_ema200
        
        # Exit when price returns to the 12h EMA20 (mean reversion to middle)
        exit_long = position == 1 and price_close < ema_20[i]
        exit_short = position == -1 and price_close > ema_20[i]
        
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

# Hypothesis: Buy when 12h closes above upper Keltner Channel with volume confirmation and price above 200 EMA; sell when 12h closes below lower Keltner Channel with volume confirmation and price below 200 EMA. Uses 1d EMA200 as trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend. Low-frequency design targets 12-37 trades/year to minimize fee drag.