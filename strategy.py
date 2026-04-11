#!/usr/bin/env python3
# 12h_1d_cci_reversion_v1
# Strategy: 12h CCI mean reversion with 1d trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: CCI > +100 indicates overbought conditions in a bearish trend (short),
# CCI < -100 indicates oversold conditions in a bullish trend (long). Combined with
# 1d EMA trend filter and volume confirmation to avoid false signals. Designed for
# low trade frequency (~15-30/year) to minimize fee drift in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_reversion_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate CCI(20) on 12h data
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(mad[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Trend filter from 1d EMA50
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # CCI signals: >100 overbought, <-100 oversold
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        
        # Entry conditions
        # Long: CCI oversold AND bullish trend AND volume confirmation
        if cci_oversold and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: CCI overbought AND bearish trend AND volume confirmation
        elif cci_overbought and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral zone (-100 to 100)
        elif position == 1 and cci[i] >= -100:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci[i] <= 100:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals