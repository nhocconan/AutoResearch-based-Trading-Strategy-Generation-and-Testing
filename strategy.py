#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 12h EMA trend filter and volume spike confirmation.
Long when %R crosses above -80 from below with volume > 1.5x average and price > 12h EMA50;
Short when %R crosses below -20 from above with volume > 1.5x average and price < 12h EMA50.
Exit when %R crosses opposite threshold (-20 for long, -80 for short) or 2x ATR stop.
Designed for ~25-35 trades/year to minimize fee drag while capturing mean reversals in trends.
Works in ranging markets via reversals and in trending markets via pullbacks to EMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    williams_r = -100 * ((highest_high - close) / rr)
    
    # 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from below with volume spike and above 12h EMA50
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
                volume[i] > 1.5 * vol_ma_20[i] and
                price_close > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from above with volume spike and below 12h EMA50
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
                  volume[i] > 1.5 * vol_ma_20[i] and
                  price_close < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R crosses opposite threshold or 2x ATR stop
            exit_signal = False
            
            if position == 1:
                # Exit long: %R crosses below -20 OR price < entry - 2*ATR
                if williams_r[i] < -20 and williams_r[i-1] >= -20:
                    exit_signal = True
                else:
                    # Track entry approximation: use price at signal as entry
                    entry_price = prices['close'].iloc[i-1] if i >= 1 else prices['close'].iloc[0]
                    if price_close < entry_price - 2.0 * atr[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: %R crosses above -80 OR price > entry + 2*ATR
                if williams_r[i] > -80 and williams_r[i-1] <= -80:
                    exit_signal = True
                else:
                    # Track entry approximation: use price at signal as entry
                    entry_price = prices['close'].iloc[i-1] if i >= 1 else prices['close'].iloc[0]
                    if price_close > entry_price + 2.0 * atr[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_12hEMA50_Trend_Volume1.5x"
timeframe = "4h"
leverage = 1.0