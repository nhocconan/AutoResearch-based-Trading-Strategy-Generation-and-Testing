#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1w = np.zeros_like(tr)
    atr_1w[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Align weekly indicators to daily timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_21_aligned[i]) or np.isnan(atr_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price_close = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: Price above weekly EMA(21) + volatility expansion
            if (price_close > ema_21_aligned[i] and 
                prices['high'].iloc[i] - prices['low'].iloc[i] > 1.5 * atr_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Price below weekly EMA(21) + volatility expansion
            elif (price_close < ema_21_aligned[i] and 
                  prices['high'].iloc[i] - prices['low'].iloc[i] > 1.5 * atr_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Price crosses back to weekly EMA(21)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below weekly EMA(21)
                if price_close < ema_21_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above weekly EMA(21)
                if price_close > ema_21_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA21_VolExp_Trend"
timeframe = "1d"
leverage = 1.0