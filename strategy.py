#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
Longs when RSI<30 and 4h close > 4h EMA50 and volume > 1.2x average.
Shorts when RSI>70 and 4h close < 4h EMA50 and volume > 1.2x average.
Exit when RSI crosses back to neutral (40-60) or 2x ATR stop.
Designed for 15-30 trades/year to minimize fee drift while capturing reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for EMA and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 50-period EMA on 4h
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate RSI on 1h
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.2x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (14-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 4h EMA to 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_50 = ema_50_4h_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: RSI oversold + 4h uptrend + volume
            if (rsi_val < 30 and 
                price_close > ema_50 and 
                vol_ratio_val > 1.2):
                signals[i] = 0.20
                position = 1
            # Enter short: RSI overbought + 4h downtrend + volume
            elif (rsi_val > 70 and 
                  price_close < ema_50 and 
                  vol_ratio_val > 1.2):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: RSI mean reversion OR ATR-based stoploss
            exit_signal = False
            
            # RSI mean reversion exit
            if position == 1 and rsi_val > 40:
                exit_signal = True
            elif position == -1 and rsi_val < 60:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry)
            if position == 1:
                # Track entry price approximation (using close at entry)
                if price_close < ema_50 - 2.0 * atr_val:  # simplified: below trend + 2x ATR
                    exit_signal = True
            elif position == -1:
                if price_close > ema_50 + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_Trend_Volume1.2x"
timeframe = "1h"
leverage = 1.0