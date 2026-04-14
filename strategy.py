# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h RSI filter and volume confirmation
# Williams %R(14) < -80 indicates oversold conditions for long entries
# Williams %R(14) > -20 indicates overbought conditions for short entries
# 12h RSI(14) > 50 filters for bullish bias in longs, < 50 for bearish bias in shorts
# Volume > 1.3x average confirms momentum behind the move
# Works in bull/bear as 12h RSI adapts to trend while Williams %R captures short-term extremes
# Target: 25-35 trades/year per symbol (100-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h RSI(14) for trend filter
    rsi_len = 14
    if len(df_12h) < rsi_len:
        return np.zeros(n)
    
    # Calculate RSI on 12h close
    delta = pd.Series(df_12h['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_12h = (100 - (100 / (1 + rs))).values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Williams %R(14) on 6h
    wr_len = 14
    highest_high = pd.Series(high).rolling(window=wr_len, min_periods=wr_len).max()
    lowest_low = pd.Series(low).rolling(window=wr_len, min_periods=wr_len).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    wr = wr.replace([np.inf, -np.inf], np.nan).fillna(0).values  # Handle division by zero
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, wr_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr[i]) or 
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = wr[i] < -80
        overbought = wr[i] > -20
        
        # 12h RSI trend filter
        bullish_trend = rsi_12h_aligned[i] > 50
        bearish_trend = rsi_12h_aligned[i] < 50
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Williams %R oversold + bullish 12h trend + volume
            if oversold and bullish_trend and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought + bearish 12h trend + volume
            elif overbought and bearish_trend and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns above -50 or 12h RSI turns bearish
            if wr[i] > -50 or rsi_12h_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns below -50 or 12h RSI turns bullish
            if wr[i] < -50 or rsi_12h_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_WilliamsR_RSI_Volume_v1"
timeframe = "6h"
leverage = 1.0