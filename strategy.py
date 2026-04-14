#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band breakout with 12-hour EMA trend filter and volume confirmation
# Long when price closes above upper Bollinger Band with volume >1.5x 20-period average and price above 12h EMA50
# Short when price closes below lower Bollinger Band with volume >1.5x 20-period average and price below 12h EMA50
# Exit when price crosses the Bollinger Band middle (SMA20)
# Bollinger Bands provide adaptive volatility-based channels, EMA50 filters trend direction, volume confirms breakout strength
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h Bollinger Bands (20-period, 2 std dev)
    close_4h = df_4h['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bollinger_upper = sma_20 + (2 * std_20)
    bollinger_lower = sma_20 - (2 * std_20)
    bollinger_middle = sma_20
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    bollinger_upper_aligned = align_htf_to_ltf(prices, df_4h, bollinger_upper)
    bollinger_lower_aligned = align_htf_to_ltf(prices, df_4h, bollinger_lower)
    bollinger_middle_aligned = align_htf_to_ltf(prices, df_4h, bollinger_middle)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 20-period calculations and EMA50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bollinger_upper_aligned[i]) or np.isnan(bollinger_lower_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: close above Bollinger upper with volume confirmation and price above 12h EMA50
            if (price > bollinger_upper_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                price > ema_50_12h_aligned[i]):                 # Price above 12h EMA50 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: close below Bollinger lower with volume confirmation and price below 12h EMA50
            elif (price < bollinger_lower_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                  price < ema_50_12h_aligned[i]):                 # Price below 12h EMA50 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Bollinger middle
            if price < bollinger_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Bollinger middle
            if price > bollinger_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0