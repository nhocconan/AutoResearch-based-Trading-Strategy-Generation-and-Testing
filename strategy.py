#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h trend filter and volume confirmation
# - 12h EMA(34) defines trend direction (long when price > EMA34, short when price < EMA34)
# - 4h RSI(14) for entry timing: long when RSI < 30 in uptrend, short when RSI > 70 in downtrend
# - Volume confirmation: current 4h volume > 1.5x 20-period average
# - Exit on opposite RSI extreme (RSI > 70 for long, RSI < 30 for short) or trend reversal
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-50 trades/year to avoid excessive fee drag

name = "4h_EMA34_RSI_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 4h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 4h volume average (20-period)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma_4h[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x average
        volume_filter = vol_ma_4h[i] > 0 and volume[i] > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 12h EMA34) + oversold RSI + volume
            if close[i] > ema_34_12h_aligned[i] and rsi_values[i] < 30 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 12h EMA34) + overbought RSI + volume
            elif close[i] < ema_34_12h_aligned[i] and rsi_values[i] > 70 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought RSI or trend reversal
            if rsi_values[i] > 70 or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold RSI or trend reversal
            if rsi_values[i] < 30 or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals