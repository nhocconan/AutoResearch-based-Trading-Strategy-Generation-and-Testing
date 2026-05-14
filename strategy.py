#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Supertrend for direction and 1h RSI(14) with volume spike for entry timing.
# Long when 4h Supertrend is bullish AND 1h RSI crosses above 30 from below AND 1h volume > 1.5 * 20-period average volume.
# Short when 4h Supertrend is bearish AND 1h RSI crosses below 70 from above AND 1h volume > 1.5 * 20-period average volume.
# Exit when RSI crosses 50 in the opposite direction or volume condition fails.
# Uses discrete position sizing (0.20) to limit fee churn. Designed for 1h timeframe with strict entry conditions.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h.

name = "1h_Supertrend_RSI_Volume_Entry_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Supertrend for trend direction (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = pd.Series(high_4h).rolling(2).max() - pd.Series(low_4h).rolling(2).min()
    tr2 = abs(pd.Series(high_4h).shift(1) - pd.Series(close_4h).shift(1))
    tr3 = abs(pd.Series(low_4h).shift(1) - pd.Series(close_4h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.full_like(close_4h, np.nan)
    direction = np.full_like(close_4h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    supertrend[atr_period] = upper_band[atr_period]
    direction[atr_period] = 1
    
    for i in range(atr_period + 1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_4h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        elif direction[i] == -1:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend direction to 1h timeframe
    supertrend_direction = direction  # 1 for uptrend, -1 for downtrend
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, supertrend_direction.astype(float))
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 1h volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(rsi_values[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 4h Supertrend bullish AND RSI crosses above 30 from below AND volume confirmation
            if (supertrend_direction_aligned[i] > 0.5 and  # Bullish trend
                rsi_values[i-1] <= 30 and rsi_values[i] > 30 and  # RSI crosses above 30
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h Supertrend bearish AND RSI crosses below 70 from above AND volume confirmation
            elif (supertrend_direction_aligned[i] < -0.5 and  # Bearish trend
                  rsi_values[i-1] >= 70 and rsi_values[i] < 70 and  # RSI crosses below 70
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below 50 from above OR volume condition fails
            if (rsi_values[i-1] >= 50 and rsi_values[i] < 50) or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI crosses above 50 from below OR volume condition fails
            if (rsi_values[i-1] <= 50 and rsi_values[i] > 50) or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals