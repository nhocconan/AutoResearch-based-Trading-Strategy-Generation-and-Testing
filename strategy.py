#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h Supertrend for direction and 1d RSI for mean reversion
# Supertrend identifies trend direction with built-in volatility filtering
# RSI identifies overbought/oversold conditions for counter-trend entries during pullbacks
# Works in both bull and bear markets: follows trend in strong moves, mean reverts in ranges
# Uses 4h for signal direction (reduces whipsaw), 1h only for entry timing
# Session filter (08-20 UTC) reduces noise trades outside active hours
# Target: 60-150 total trades over 4 years = 15-37/year for 1h

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 4h data ONCE for Supertrend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Supertrend (10, 3.0) on 4h
    atr_period = 10
    atr_mult = 3.0
    
    # True Range
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Basic Upper and Lower Bands
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl2 + (atr_mult * atr)
    lower_band = hl2 - (atr_mult * atr)
    
    # Final Upper and Lower Bands
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    
    for i in range(1, len(df_4h)):
        if df_4h['close'].iloc[i] <= final_upper.iloc[i-1]:
            final_upper.iloc[i] = min(final_upper.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i] = upper_band.iloc[i]
            
        if df_4h['close'].iloc[i] >= final_lower.iloc[i-1]:
            final_lower.iloc[i] = max(final_lower.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i] = lower_band.iloc[i]
    
    # Supertrend direction
    supertrend = np.zeros(len(df_4h))
    for i in range(len(df_4h)):
        if i == 0:
            supertrend[i] = 1
        else:
            if supertrend[i-1] == -1 and df_4h['close'].iloc[i] > final_upper.iloc[i]:
                supertrend[i] = 1
            elif supertrend[i-1] == 1 and df_4h['close'].iloc[i] < final_lower.iloc[i]:
                supertrend[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
    
    # Align Supertrend to 1h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend.values)
    
    # Load 1d data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI (14) on 1d
    rsi_period = 14
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when insufficient data
    
    # Align RSI to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(30, 14)  # Need enough for RSI and Supertrend
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        st_direction = supertrend_aligned[i]
        rsi_value = rsi_aligned[i]
        
        if position == 0:
            # Enter long: uptrend + RSI not overbought (< 60)
            if st_direction == 1 and rsi_value < 60:
                position = 1
                signals[i] = position_size
            # Enter short: downtrend + RSI not oversold (> 40)
            elif st_direction == -1 and rsi_value > 40:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend reversal OR RSI overbought (>= 70)
            if st_direction == -1 or rsi_value >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend reversal OR RSI oversold (<= 30)
            if st_direction == 1 or rsi_value <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hSupertrend_1dRSI_Pullback_v1"
timeframe = "1h"
leverage = 1.0