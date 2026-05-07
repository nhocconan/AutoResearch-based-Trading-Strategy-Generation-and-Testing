#!/usr/bin/env python3
name = "6h_1d_PriceAction_Reversal_With_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 14-period RSI
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # 6h price action: check for pin bar patterns
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    
    # Bullish pin bar: long lower wick, small body, close near high
    bullish_pin = (lower_wick > 2 * body_size) & (body_size < (high - low) * 0.3) & (close > open_)
    # Bearish pin bar: long upper wick, small body, close near low
    bearish_pin = (upper_wick > 2 * body_size) & (body_size < (high - low) * 0.3) & (close < open_)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 4)  # Wait for RSI and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_4[i]) or 
            np.isnan(bullish_pin[i]) or np.isnan(bearish_pin[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish pin bar at RSI oversold with volume confirmation
            if bullish_pin[i] and rsi_1d_aligned[i] < 30 and volume[i] > vol_ma_4[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # Short: bearish pin bar at RSI overbought with volume confirmation
            elif bearish_pin[i] and rsi_1d_aligned[i] > 70 and volume[i] > vol_ma_4[i] * 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral or opposite pin bar appears
            if rsi_1d_aligned[i] > 50 or bearish_pin[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral or opposite pin bar appears
            if rsi_1d_aligned[i] < 50 or bullish_pin[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h price action reversal with daily RSI extremes and volume confirmation
# - Uses daily RSI to identify overbought/oversold conditions on higher timeframe
# - Looks for pin bar reversals (strong rejection) on 6h chart at RSI extremes
# - Volume confirmation (1.5x average) ensures institutional participation
# - Works in both bull (buy oversold pins in rallies) and bear (sell overbought pins in declines)
# - Exits when RSI returns to neutral (50) or opposite signal appears
# - Position size 0.25 targets ~20-60 trades/year, avoiding fee drag
# - Price action based, avoids lagging indicators
# - Designed to work in ranging and trending markets via RSI filter