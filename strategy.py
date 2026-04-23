#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R crosses above -80 from oversold AND close > 1d EMA50 AND volume > 1.3x 20-period average.
Short when Williams %R crosses below -20 from overbought AND close < 1d EMA50 AND volume > 1.3x 20-period average.
Exit when Williams %R crosses -50 (mean reversion) or opposing signal occurs.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Williams %R is a momentum oscillator that identifies overbought/oversold conditions, effective in ranging markets.
The 1d EMA50 ensures alignment with higher timeframe trend, reducing counter-trend trades.
Volume confirmation adds conviction to reversals.
"""

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
    
    # Load 6h data for Williams %R calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (already on 6h, but align for safety)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else -50
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold AND close > 1d EMA50 AND volume spike
            if (wr > -80 and wr_prev <= -80 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -20 from overbought AND close < 1d EMA50 AND volume spike
            elif (wr < -20 and wr_prev >= -20 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (mean reversion) or price < EMA50
                if wr > -50 or close[i] < ema50_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (mean reversion) or price > EMA50
                if wr < -50 or close[i] > ema50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0