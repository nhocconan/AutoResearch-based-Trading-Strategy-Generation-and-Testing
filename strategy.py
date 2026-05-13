#!/usr/bin/env python3
# Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. Long when %R < -80 (oversold) and price > 1d EMA34 with volume spike.
# Short when %R > -20 (overbought) and price < 1d EMA34 with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 20-50 trades/year by requiring strict confluence of momentum, trend, and volume.
# Works in bull markets via oversold bounces and in bear markets via overbought reversals.

name = "4h_WilliamsR_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams %R(14) on primary timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_spike[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume spike
            if williams_r[i] < -80 and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume spike
            elif williams_r[i] > -20 and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (momentum weakening) OR price < 1d EMA34 (trend break)
            if williams_r[i] > -50 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (momentum weakening) OR price > 1d EMA34 (trend break)
            if williams_r[i] < -50 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals