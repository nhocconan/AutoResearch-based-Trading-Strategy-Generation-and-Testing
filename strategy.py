#!/usr/bin/env python3
# 4H_Multi_Timeframe_Momentum_Combo
# Hypothesis: Combines momentum from multiple timeframes (4h price action, 1d RSI, and 12h trend) with volume confirmation.
# Long when: 4h close > 4h open (bullish candle) AND 1d RSI > 50 (bullish momentum) AND 12h EMA20 trending up AND volume > 1.5x average.
# Short when: 4h close < 4h open (bearish candle) AND 1d RSI < 50 (bearish momentum) AND 12h EMA20 trending down AND volume > 1.5x average.
# Uses discrete position sizing (0.25) to minimize churn and targets 20-40 trades/year.
# Designed to work in both bull and bear markets by requiring alignment across multiple timeframes.

name = "4H_Multi_Timeframe_Momentum_Combo"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 12h EMA(20) with proper initialization
    ema_20_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 20:
        ema_20_12h[19] = np.mean(close_12h[0:20])
        for i in range(20, len(close_12h)):
            ema_20_12h[i] = (close_12h[i] * 2 + ema_20_12h[i-1] * 18) / 20
    
    # Align 12h EMA to 4h timeframe
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (gain[i] + avg_gain[i-1] * 13) / 14
            avg_loss[i] = (loss[i] + avg_loss[i-1] * 13) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_1d = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # Align daily RSI to 4h timeframe
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_20_12h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine candle direction
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Enter long: Bullish candle AND bullish RSI (>50) AND uptrend (price > EMA20) AND volume confirmation
            if bullish_candle and rsi_14_1d_aligned[i] > 50 and close[i] > ema_20_12h_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish candle AND bearish RSI (<50) AND downtrend (price < EMA20) AND volume confirmation
            elif bearish_candle and rsi_14_1d_aligned[i] < 50 and close[i] < ema_20_12h_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish candle OR RSI turns bearish (<50) OR trend turns down (price < EMA20)
            if bearish_candle or rsi_14_1d_aligned[i] < 50 or close[i] < ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish candle OR RSI turns bullish (>50) OR trend turns up (price > EMA20)
            if bullish_candle or rsi_14_1d_aligned[i] > 50 or close[i] > ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals