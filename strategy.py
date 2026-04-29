#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# Uses RSI(14) on 1h for oversold/overbought signals, filtered by 4h EMA50 trend direction.
# Only takes long positions when RSI < 30 and price > 4h EMA50 (uptrend pullback).
# Only takes short positions when RSI > 70 and price < 4h EMA50 (downtrend bounce).
# Volume confirmation (>1.5x 20-period average) ensures momentum behind the move.
# Designed for ~20-40 trades/year on 1h timeframe to minimize fee drag while capturing high-probability mean reversion in trending markets.
# Works in both bull and bear markets via 4h trend filter - only trades mean reversion in trend direction.

name = "1h_RSI_MeanReversion_4hEMA50_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 20-period average volume for confirmation (on 1h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # RSI and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi_values[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (50) or stoploss via signal=0
            if curr_rsi >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) or stoploss via signal=0
            if curr_rsi <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: RSI oversold (<30) in uptrend (price > 4h EMA50) with volume confirmation
            if vol_confirm and curr_rsi < 30 and curr_close > curr_ema50_4h:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry: RSI overbought (>70) in downtrend (price < 4h EMA50) with volume confirmation
            elif vol_confirm and curr_rsi > 70 and curr_close < curr_ema50_4h:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals