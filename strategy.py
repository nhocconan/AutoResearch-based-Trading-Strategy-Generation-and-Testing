#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R (14) extreme reversal with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.5x average.
Short when Williams %R crosses below -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.5x average.
Exit when Williams %R returns to neutral zone (-50) or trend reverses (price crosses 1d EMA50).
Williams %R is effective in ranging markets and captures reversals; EMA50 filter ensures we only trade with the higher timeframe trend.
Targets 20-40 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 for 1d trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Williams %R (14) on 4h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND price > 1d EMA50 (uptrend) AND volume spike
            if (wr > -80 and williams_r[i-1] <= -80 and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND price < 1d EMA50 (downtrend) AND volume spike
            elif (wr < -20 and williams_r[i-1] >= -20 and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to neutral zone (-50) OR price breaks below 1d EMA50 (trend reversal)
                if wr >= -50 or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to neutral zone (-50) OR price breaks above 1d EMA50 (trend reversal)
                if wr <= -50 or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_1dEMA50_Volume_Reversal"
timeframe = "4h"
leverage = 1.0