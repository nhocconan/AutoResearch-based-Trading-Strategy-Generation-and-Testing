#!/usr/bin/env python3
# Hypothesis: 4h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND close > 1d EMA50 AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND close < 1d EMA50 AND volume > 1.5x average
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR trend reversal.
# Uses 4h timeframe with 1d trend filter for BTC/ETH resilience in both bull and bear markets.
# Williams %R identifies exhaustion points; 1d EMA50 filters trend; volume confirms participation.

name = "4h_WilliamsR_1dEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 4h data (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1d data for EMA50 trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 4h volume > 1.5x 20-period average (participation confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND close > 1d EMA50 AND volume participation
            if williams_r[i] < -80 and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND close < 1d EMA50 AND volume participation
            elif williams_r[i] > -20 and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (recovery) OR trend reversal (close < 1d EMA50)
            if williams_r[i] > -50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (decline) OR trend reversal (close > 1d EMA50)
            if williams_r[i] < -50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals