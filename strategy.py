#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d CCI trend filter with volume confirmation.
# Williams %R(14) identifies oversold/overbought conditions for mean reversion entries.
# 1d CCI(20) > 100 = bullish trend, < -100 = bearish trend - provides directional bias.
# Volume > 1.3x average confirms institutional participation.
# Works in bull/bear markets as 1d CCI adapts to long-term trend.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d CCI(20) for trend filter
    cci_len = 20
    if len(df_1d) < cci_len:
        return np.zeros(n)
    
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma_tp = typical_price.rolling(window=cci_len, min_periods=cci_len).mean()
    mad = typical_price.rolling(window=cci_len, min_periods=cci_len).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_1d = (typical_price - sma_tp) / (0.015 * mad)
    cci_1d_values = cci_1d.values
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d_values)
    
    # Williams %R(14) on 4h
    wr_len = 14
    highest_high = pd.Series(high).rolling(window=wr_len, min_periods=wr_len).max()
    lowest_low = pd.Series(low).rolling(window=wr_len, min_periods=wr_len).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    wr_values = wr.values
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(40, wr_len, cci_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_values[i]) or 
            np.isnan(cci_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d CCI indicates trend direction
        bullish_trend = cci_1d_aligned[i] > 100
        bearish_trend = cci_1d_aligned[i] < -100
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + bullish trend + volume
            if (wr_values[i] < -80 and 
                bullish_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought (> -20) + bearish trend + volume
            elif (wr_values[i] > -20 and 
                  bearish_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to overbought (> -20) or trend turns bearish
            if wr_values[i] > -20 or cci_1d_aligned[i] < -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to oversold (< -80) or trend turns bullish
            if wr_values[i] < -80 or cci_1d_aligned[i] > 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_CCI_Volume_v1"
timeframe = "4h"
leverage = 1.0