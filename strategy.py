#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily 10-period EMA for trend filter ===
    close_1d = df_1d['close'].values
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    
    # === Daily RSI(14) for mean reversion signals ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_10_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_10_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above EMA (uptrend) + RSI oversold + volume confirmation
            if (price_close > ema_trend and 
                rsi_val < 30 and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA (downtrend) + RSI overbought + volume confirmation
            elif (price_close < ema_trend and 
                  rsi_val > 70 and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse RSI extreme or trend change
            if position == 1 and (rsi_val > 70 or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_val < 30 or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4d_EMA10_RSI_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0