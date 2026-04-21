#!/usr/bin/env python3
"""
Hypothesis: 1h EMA pullback in 4h trend with volume confirmation and session filter.
Trades only during 08-20 UTC to avoid low-volume Asian session noise.
Long when price pulls back to 21 EMA in 4h uptrend (EMA50 > EMA200) with volume > 1.5x average.
Short when price pulls back to 21 EMA in 4h downtrend (EMA50 < EMA200) with volume > 1.5x average.
Exit on EMA crossover or 2x ATR stop. Designed for 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 and EMA200 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1h EMA21 for entry timing
    close = prices['close'].values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume spike > 1.5x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (14-period)
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(ema21[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema50 = ema50_4h_aligned[i]
        ema200 = ema200_4h_aligned[i]
        ema21_val = ema21[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: 4h uptrend + price at 21 EMA + volume
            if (ema50 > ema200 and  # 4h uptrend
                price_close >= ema21_val * 0.998 and  # near 21 EMA (allow 0.2% slack)
                price_close <= ema21_val * 1.002 and
                vol_ratio_val > 1.5):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + price at 21 EMA + volume
            elif (ema50 < ema200 and  # 4h downtrend
                  price_close >= ema21_val * 0.998 and  # near 21 EMA
                  price_close <= ema21_val * 1.002 and
                  vol_ratio_val > 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: 4h EMA crossover OR 2x ATR stop
            exit_signal = False
            
            # EMA crossover exit
            if position == 1 and ema50 < ema200:
                exit_signal = True
            elif position == -1 and ema50 > ema200:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry approximated by 21 EMA)
            if position == 1:
                if price_close < ema21_val - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                if price_close > ema21_val + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA21_Pullback_4hEMA50_200_Trend_Volume1.5x_Session"
timeframe = "1h"
leverage = 1.0