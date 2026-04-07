#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion on Bollinger Bands with 4h trend filter
# Long when price touches lower BB(20,2) in 4h uptrend (4h close > EMA20)
# Short when price touches upper BB(20,2) in 4h downtrend (4h close < EMA20)
# Exit when price crosses middle BB (20-period SMA)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses Bollinger Bands from 1h and trend filter from 4h EMA
# Target: 75-200 total trades over 4 years (19-50/year)

name = "1h_bb_mean_reversion_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Bollinger Bands (20, 2) on 1h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    middle_band = sma_20
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(sma_20[i]):
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
            # Exit: price crosses above middle Bollinger Band
            elif close[i] > middle_band[i]:
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
            # Exit: price crosses below middle Bollinger Band
            elif close[i] < middle_band[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Trend filter: 4h EMA20
            uptrend = close[i] > ema_20_4h_aligned[i]
            downtrend = close[i] < ema_20_4h_aligned[i]
            
            # Long: price touches lower Bollinger Band in uptrend
            if close[i] <= lower_band[i] and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price touches upper Bollinger Band in downtrend
            elif close[i] >= upper_band[i] and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals