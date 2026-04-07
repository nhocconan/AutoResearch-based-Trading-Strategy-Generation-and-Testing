#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with 1-day trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) in uptrend (price > 1-day EMA50) with volume > 1.5x average
# Short when Williams %R > -20 (overbought) in downtrend (price < 1-day EMA50) with volume > 1.5x average
# Exit when Williams %R crosses -50 or trend reverses
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25
# Uses 1-day EMA50 for trend direction and Williams %R for timing
# Target: 100-150 total trades over 4 years (25-38/year)

name = "4h_williamsr_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R(14) on 4h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 4-period volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=4, min_periods=4).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
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
            # Exit: Williams %R crosses -50 (mean reversion) or trend fails
            elif williams_r[i] > -50 or close[i] <= ema50_1d_aligned[i]:
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
            # Exit: Williams %R crosses -50 or trend fails
            elif williams_r[i] < -50 or close[i] >= ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme with volume confirmation and trend alignment
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            
            # Long: Williams %R < -80 (oversold), volume spike, uptrend (price > EMA50)
            if (williams_r[i] < -80 and volume_spike and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R > -20 (overbought), volume spike, downtrend (price < EMA50)
            elif (williams_r[i] > -20 and volume_spike and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals