# 4h_MultiTf_RSI_CCI_Strategy
# Multi-timeframe strategy using 4h RSI and 1d CCI for trend confirmation
# Entry: RSI(14) > 55 and CCI(20) > 100 for long, opposite for short
# Exit: RSI crosses back below 50 (long) or above 50 (short)
# Volume filter: current volume > 20-period average
# Position size: 0.25 to limit drawdown
# Designed for 4h timeframe with 1d trend filter to work in both bull and bear markets
# Target: 20-50 trades per year to avoid excessive fee drag

#!/usr/bin/env python3
name = "4h_MultiTf_RSI_CCI_Strategy"
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
    
    # Load 1D data ONCE for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CCI(20) on daily data
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    tp_series = pd.Series(typical_price)
    ma = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - ma.values) / (0.015 * mad.values)
    # Handle division by zero or infinite values
    cci = np.where((mad.values == 0) | np.isnan(mad.values) | np.isinf(mad.values), 0, cci)
    
    # Align CCI to 4h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # Calculate RSI(14) on 4h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    rs = avg_gain.values / np.where(avg_loss.values == 0, 1, avg_loss.values)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for indicators
        if (np.isnan(rsi[i]) or np.isnan(cci_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI > 55 and CCI > 100 with volume confirmation
            if (rsi[i] > 55) and (cci_aligned[i] > 100) and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 45 and CCI < -100 with volume confirmation
            elif (rsi[i] < 45) and (cci_aligned[i] < -100) and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals