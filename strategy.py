#!/usr/bin/env python3
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
    
    # Get daily data for ATR and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-day ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.mean(tr[i-14:i])
    
    # Align ATR to 4h timeframe (1 day = 6 * 4h bars)
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4-period RSI on 4h close prices
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(4, n):
        if i == 4:
            avg_gain[i] = np.mean(gain[0:4])
            avg_loss[i] = np.mean(loss[0:4])
        else:
            avg_gain[i] = (avg_gain[i-1] * 3 + gain[i]) / 4
            avg_loss[i] = (avg_loss[i-1] * 3 + loss[i]) / 4
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20, 2) on 4h
    sma_20 = np.full(n, np.nan)
    std_20 = np.full(n, np.nan)
    for i in range(20, n):
        sma_20[i] = np.mean(close[i-20:i])
        std_20[i] = np.std(close[i-20:i])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width for regime detection
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Calculate 20-period percentile of BB width for regime classification
    bb_width_percentile = np.full(n, np.nan)
    for i in range(40, n):  # Need 20 * 2 for percentile calculation
        window = bb_width[i-20:i]
        if not np.all(np.isnan(window)):
            # Calculate percentile of current value in window
            bb_width_percentile[i] = np.sum(~np.isnan(window) & (window <= bb_width[i])) / np.sum(~np.isnan(window)) * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(atr_14_4h[i]) or np.isnan(rsi[i]) or np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: BB width percentile > 60 = trending regime (avoid chop)
        trending_regime = bb_width_percentile[i] > 60
        
        # Volatility filter: ATR > 0.5 * price (avoid extremely low volatility)
        vol_filter = atr_14_4h[i] > 0.005 * close[i]
        
        # RSI conditions: oversold (<30) for long, overbought (>70) for short
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Bollinger Band conditions: price outside bands
        bb_break_low = close[i] < lower_bb[i]
        bb_break_high = close[i] > upper_bb[i]
        
        # Entry conditions
        long_entry = rsi_oversold and bb_break_low and trending_regime and vol_filter
        short_entry = rsi_overbought and bb_break_high and trending_regime and vol_filter
        
        # Exit conditions: opposite RSI extreme or middle BB
        long_exit = rsi[i] > 50 or close[i] > sma_20[i]
        short_exit = rsi[i] < 50 or close[i] < sma_20[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_rsi_bb_width_regime_v1"
timeframe = "4h"
leverage = 1.0