#!/usr/bin/env python3
"""
Hypothesis: 1d Exponential Moving Average (EMA) crossover with 1w trend filter and volume confirmation.
Long when 21-period EMA crosses above 55-period EMA with weekly close above weekly 50-EMA and volume > 1.5x average.
Short when 21-period EMA crosses below 55-period EMA with weekly close below weekly 50-EMA and volume > 1.5x average.
Exit when EMA cross reverses or 2x ATR stoploss is hit.
Designed for low-frequency signals (target 15-30 trades/year) to minimize fee drag while capturing major trends.
Works in bull markets via trend following and in bear markets via short signals with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly 50-period EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily EMAs for entry signal
    close = prices['close'].values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Calculate EMA crossover signals
    ema_cross_up = (ema_21 > ema_55) & (np.roll(ema_21, 1) <= np.roll(ema_55, 1))
    ema_cross_down = (ema_21 < ema_55) & (np.roll(ema_21, 1) >= np.roll(ema_55, 1))
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_21[i]) or np.isnan(ema_55[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema21_val = ema_21[i]
        ema55_val = ema_55[i]
        ema50_1w_val = ema_50_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: EMA21 crosses above EMA55 with weekly trend and volume
            if (ema_cross_up[i] and 
                price_close > ema50_1w_val and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: EMA21 crosses below EMA55 with weekly trend and volume
            elif (ema_cross_down[i] and 
                  price_close < ema50_1w_val and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: EMA cross reverses OR ATR-based stoploss
            exit_signal = False
            
            # EMA cross reversal exit
            if position == 1 and ema_cross_down[i]:
                exit_signal = True
            elif position == -1 and ema_cross_up[i]:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry approximation)
            if position == 1:
                # For longs, stop below EMA55 (as proxy for entry area)
                if price_close < ema55_val - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above EMA21 (as proxy for entry area)
                if price_close > ema21_val + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_EMA21_55_Crossover_1wEMA50_Trend_Volume1.5x_ATR2x"
timeframe = "1d"
leverage = 1.0