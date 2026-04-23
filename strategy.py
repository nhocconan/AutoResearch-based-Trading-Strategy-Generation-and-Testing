#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Reversal with 1d EMA50 trend filter and volume spike confirmation.
Long when Williams %R crosses above -80 (oversold) AND close > 1d EMA50 AND volume > 1.8x 20-period average.
Short when Williams %R crosses below -20 (overbought) AND close < 1d EMA50 AND volume > 1.8x 20-period average.
Exit when Williams %R reverts to -50 (mean) or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Williams %R provides reversal signals that work in both bull and bear markets by capturing exhaustion.
1d EMA50 provides long-term trend filter that avoids counter-trend trades in strong trends.
Volume spike confirms institutional participation in the reversal.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Williams %R calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Williams %R on 4h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 4h Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        williams_r_val = williams_r_aligned[i]
        williams_r_prev = williams_r_aligned[i-1] if i > 0 else williams_r_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND close > 1d EMA50 AND volume spike
            if (williams_r_prev <= -80 and williams_r_val > -80 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -20 (overbought) AND close < 1d EMA50 AND volume spike
            elif (williams_r_prev >= -20 and williams_r_val < -20 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R reverts to -50 or ATR stoploss
                if williams_r_val >= -50:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R reverts to -50 or ATR stoploss
                if williams_r_val <= -50:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0