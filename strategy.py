#!/usr/bin/env python3
# 4h_12h_ema_cross_volume_v1
# Hypothesis: 4h EMA crossover with 12h EMA trend filter and volume confirmation.
# Long: 4h EMA(9) crosses above EMA(21) AND price > 12h EMA(50) AND volume > 1.5x 20-period average.
# Short: 4h EMA(9) crosses below EMA(21) AND price < 12h EMA(50) AND volume > 1.5x 20-period average.
# Exit: Opposite EMA cross or ATR trailing stop (2.0x ATR from extreme).
# Uses 12h EMA for major trend filter, 4h for execution timing, volume for confirmation, ATR for dynamic stops.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_ema_cross_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA(9) and EMA(21) for crossover signal
    close_s = pd.Series(close)
    ema9 = close_s.ewm(span=9, min_periods=9, adjust=False).mean().values
    ema21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Get 12h data for EMA(50) trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align HTF EMA50 to 4h timeframe (wait for completed 12h bar)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # EMA crossover signals
        ema9_prev = ema9[i-1] if i > 0 else ema9[i]
        ema21_prev = ema21[i-1] if i > 0 else ema21[i]
        bullish_cross = (ema9[i] > ema21[i]) and (ema9_prev <= ema21_prev)
        bearish_cross = (ema9[i] < ema21[i]) and (ema9_prev >= ema21_prev)
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if long_high > 0 and close[i] < long_high - 2.0 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Bearish EMA crossover
            elif bearish_cross:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if short_low > 0 and close[i] > short_low + 2.0 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Bullish EMA crossover
            elif bullish_cross:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for EMA crossover with volume confirmation and 12h trend filter
            bullish_setup = bullish_cross and volume_confirmed and (close[i] > ema50_12h_aligned[i])
            bearish_setup = bearish_cross and volume_confirmed and (close[i] < ema50_12h_aligned[i])
            
            if bullish_setup:
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals