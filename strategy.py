#!/usr/bin/env python3
"""
4H_RSI_TREND_WITH_VOLUME_CONFIRMATION
Hypothesis: Use RSI(14) with 50 level crossovers for trend direction, confirmed by volume spike (2.0x 20-period) and filtered by 4h ADX > 25 to avoid ranging markets.
Long when RSI crosses above 50 with volume spike and ADX > 25.
Short when RSI crosses below 50 with volume spike and ADX > 25.
Exit when RSI returns to 50 level or volume drops.
Designed to capture trends in both bull and bear markets with controlled trade frequency.
"""
name = "4H_RSI_TREND_WITH_VOLUME_CONFIRMATION"
timeframe = "4h"
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
    
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ADX calculation for trend strength
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift())
    tr3 = abs(pd.Series(low) - pd.Series(close).shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        if (np.isnan(rsi_values[i]) or 
            np.isnan(adx_values[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI crosses above 50 with volume spike and strong trend
            if (rsi_values[i] > 50 and rsi_values[i-1] <= 50 and 
                volume_spike[i] and adx_values[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below 50 with volume spike and strong trend
            elif (rsi_values[i] < 50 and rsi_values[i-1] >= 50 and 
                  volume_spike[i] and adx_values[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to 50 or volume drops
            if rsi_values[i] <= 50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to 50 or volume drops
            if rsi_values[i] >= 50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals