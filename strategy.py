#!/usr/bin/env python3
# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and volume confirmation.
# Uses 4h EMA200 for trend alignment (HTF), 1h RSI(2) for oversold/overbought entries, and volume spike (>1.5x 20-bar avg) for confirmation.
# Designed for low trade frequency (target 60-150 total over 4 years) to minimize fee drag while capturing mean reversion in trends.
# Works in both bull and bear markets by only taking mean reversion trades in the direction of the 4h trend.

name = "1h_RSI2_MeanReversion_4hEMA200_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA200 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate RSI(2) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # fill NaN with 50 (neutral)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # start after RSI lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_200_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI(2) < 10 (oversold), price > 4h EMA200 (uptrend), volume spike (>1.5x avg)
            if (rsi[i] < 10 and 
                close[i] > ema_200_4h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI(2) > 90 (overbought), price < 4h EMA200 (downtrend), volume spike (>1.5x avg)
            elif (rsi[i] > 90 and 
                  close[i] < ema_200_4h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if RSI(2) > 50 (mean reversion complete) or volume drops
            if (rsi[i] > 50) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close position if RSI(2) < 50 (mean reversion complete) or volume drops
            if (rsi[i] < 50) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals