#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly EMA(21) for trend filter ===
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === Daily 20-period EMA for entry filter ===
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === Daily RSI(14) for momentum filter ===
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_vals = rsi_14.values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(rsi_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_21_1w_val = ema_21_1w_aligned[i]
        ema_20_1d_val = ema_20_1d_aligned[i]
        rsi_val = rsi_14_aligned[i]
        
        if position == 0:
            # Enter long: price above weekly EMA21, above daily EMA20, RSI > 50
            if (price_close > ema_21_1w_val and 
                price_close > ema_20_1d_val and 
                rsi_val > 50):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly EMA21, below daily EMA20, RSI < 50
            elif (price_close < ema_21_1w_val and 
                  price_close < ema_20_1d_val and 
                  rsi_val < 50):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse condition
            if position == 1 and (price_close < ema_20_1d_val or rsi_val < 40):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > ema_20_1d_val or rsi_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA21_DailyEMA20_RSI_Filter"
timeframe = "1d"
leverage = 1.0