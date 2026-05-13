#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1w trend filter (price > 200 EMA) and volume confirmation (>1.3x avg volume). Uses ATR(14) trailing stop (2.0x) for risk control. Discrete sizing 0.25.
# Target: 60-120 total trades over 4 years (15-30/year) on 12h timeframe.
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets.
# 1w EMA200 filter ensures we only take mean reversion trades in the direction of the higher timeframe trend.
# Volume confirmation filters out low-conviction moves. Works in bull markets via buying dips in uptrend and in bear markets via selling rallies in downtrend.

name = "12h_WilliamsR_MeanReversion_1wEMA200_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(200) for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 12h timeframe (wait for 1w bar to close)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Williams %R(14) on 12h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) AND price > 1w EMA200 (uptrend filter) AND volume > 1.3x average
            if (williams_r[i] < -80 and 
                close[i] > ema_200_1w_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Williams %R overbought (> -20) AND price < 1w EMA200 (downtrend filter) AND volume > 1.3x average
            elif (williams_r[i] > -20 and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR) OR Williams %R exits oversold (> -50)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            williams_exit = williams_r[i] > -50
            if trailing_stop or williams_exit:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR) OR Williams %R exits overbought (< -50)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            williams_exit = williams_r[i] < -50
            if trailing_stop or williams_exit:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals