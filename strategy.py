#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) with 1d EMA200 filter and volume confirmation
# Long when W%R < -80 (oversold), price > 1d EMA200 (uptrend bias), and volume > 1.5x average
# Short when W%R > -20 (overbought), price < 1d EMA200 (downtrend bias), and volume > 1.5x average
# Exit when W%R crosses back above -50 (for longs) or below -50 (for shorts)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1d EMA200 for trend bias and 6h volume average for confirmation
# Target: 100-180 total trades over 4 years (25-45/year)

name = "6h_williamsr_1d_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 6h volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after 200 for EMA200
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: W%R crosses back above -50 (exit overbought condition)
            elif williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: W%R crosses back below -50 (exit oversold condition)
            elif williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with W%R extremes, trend bias, and volume confirmation
            # Long: W%R < -80 (oversold), price above 1d EMA200, volume spike
            long_condition = (williams_r[i] < -80 and 
                            close[i] > ema200_1d_aligned[i] and
                            volume[i] > 1.5 * volume_ma[i])
            
            # Short: W%R > -20 (overbought), price below 1d EMA200, volume spike
            short_condition = (williams_r[i] > -20 and 
                             close[i] < ema200_1d_aligned[i] and
                             volume[i] > 1.5 * volume_ma[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals