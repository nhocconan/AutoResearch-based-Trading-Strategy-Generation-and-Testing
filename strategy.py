#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour EMA crossover with 4-hour trend filter and volume confirmation
# Long when 1h EMA20 crosses above EMA50, 4h close > 4h EMA50 (uptrend), and volume > 1.5x 1h average volume
# Short when 1h EMA20 crosses below EMA50, 4h close < 4h EMA50 (downtrend), and volume > 1.5x 1h average volume
# Exit when EMA crossover reverses or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h EMA50 for trend filter and 1h volume average for confirmation
# Target: 100-150 total trades over 4 years (25-38/year)

name = "1h_ema_cross_4h_ema50_vol_v1"
timeframe = "1h"
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
    
    # 1h EMA20 and EMA50 for crossover
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h volume average for confirmation
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: EMA crossover reverses or trend changes
            elif ema20[i] < ema50[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: EMA crossover reverses or trend changes
            elif ema20[i] > ema50[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with EMA crossover, trend alignment, and volume confirmation
            # Bullish crossover: EMA20 crosses above EMA50
            bullish_cross = ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1]
            # Bearish crossover: EMA20 crosses below EMA50
            bearish_cross = ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1]
            
            # Long: bullish crossover, 4h uptrend, volume spike
            if (bullish_cross and
                close[i] > ema50_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: bearish crossover, 4h downtrend, volume spike
            elif (bearish_cross and
                  close[i] < ema50_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals