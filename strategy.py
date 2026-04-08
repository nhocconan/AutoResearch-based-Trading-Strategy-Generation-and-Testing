#!/usr/bin/env python3
# [24932] 12h_1d_atr_breakout_v1
# Hypothesis: 12-hour ATR breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above ATR(20) based upper band with volume > 1.5x average and close > 1-day EMA(50).
# Short when price breaks below ATR(20) based lower band with volume > 1.5x average and close < 1-day EMA(50).
# Exit when price crosses the 1-day EMA(50) or volatility drops (ATR ratio < 0.8).
# Uses volatility-based breakouts which work in both trending and ranging markets by adapting to market conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_atr_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1-day close
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema = pd.Series(close_1d).ewm(span=50, adjust=False).values
        ema_50_1d[:] = ema
    
    # Calculate ATR(20) for volatility bands
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr_values = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().values
        atr[:] = atr_values
    
    # Calculate ATR-based bands (2 * ATR)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    if n >= atr_period:
        upper_band[atr_period:] = close[atr_period:] + 2.0 * atr[atr_period:]
        lower_band[atr_period:] = close[atr_period:] - 2.0 * atr[atr_period:]
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day EMA to 12-hour timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below EMA(50) or volatility drops significantly
            if price < ema_trend or (atr[i] > 0 and atr[i] < 0.8 * atr[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above EMA(50) or volatility drops significantly
            if price > ema_trend or (atr[i] > 0 and atr[i] < 0.8 * atr[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper ATR band with volume expansion and above EMA(50)
            if price > upper_band[i] and vol_ratio > 1.5 and price > ema_trend:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower ATR band with volume expansion and below EMA(50)
            elif price < lower_band[i] and vol_ratio > 1.5 and price < ema_trend:
                position = -1
                signals[i] = -0.25
    
    return signals