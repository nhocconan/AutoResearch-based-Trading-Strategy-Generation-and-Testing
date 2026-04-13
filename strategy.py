#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 12-period RSI with 12h volume confirmation and 12h price above 50-period EMA.
# Uses 12h RSI for mean reversion signals, volume to confirm momentum, and EMA for trend filter.
# This combination avoids whipsaws by requiring alignment of momentum, volume, and trend.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12-period RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=12, min_periods=12).mean()
    avg_loss = loss.rolling(window=12, min_periods=12).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 50-period EMA for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume and its 20-period average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_values[i]) or np.isnan(ema_50[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period volume average
        volume_condition = volume[i] > (volume_ma_20[i] * 1.5)
        
        # RSI conditions: oversold for long, overbought for short
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        # Trend filter: price above EMA50 for long, below EMA50 for short
        price_above_ema = close[i] > ema_50[i]
        price_below_ema = close[i] < ema_50[i]
        
        # Entry conditions
        if position == 0:
            if rsi_oversold and volume_condition and price_above_ema:
                position = 1
                signals[i] = position_size
            elif rsi_overbought and volume_condition and price_below_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when RSI becomes overbought or price falls below EMA50
            if rsi_values[i] > 70 or close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when RSI becomes oversold or price rises above EMA50
            if rsi_values[i] < 30 or close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_RSI_Volume_EMA50_Filter_v1"
timeframe = "12h"
leverage = 1.0