#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action with daily trend filter and volume confirmation
# Long when price breaks above 12h resistance with volume spike and daily bullish trend
# Short when price breaks below 12h support with volume spike and daily bearish trend
# Exit on 12h midline cross
# Uses daily trend to avoid counter-trend trades in bear markets
# Target: 15-30 trades per symbol over 4 years (4-7.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and daily data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 12h support/resistance levels (24-period lookback for 12d equiv)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    resistance = pd.Series(high_12h).rolling(window=24, min_periods=24).max().values
    support = pd.Series(low_12h).rolling(window=24, min_periods=24).min().values
    midline = (resistance + support) / 2
    
    # Calculate daily EMA for trend filter (21-period)
    close_daily = df_daily['close'].values
    ema_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h volume average (24-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values
    
    # Align indicators to 12h timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_12h, resistance)
    support_aligned = align_htf_to_ltf(prices, df_12h, support)
    midline_aligned = align_htf_to_ltf(prices, df_12h, midline)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 24-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_current = volume[i]  # Current 12h volume
        
        if position == 0:
            # Long setup: break above resistance with volume spike and daily bullish trend
            if (price > resistance_aligned[i] and 
                vol_12h_current > 1.8 * vol_ma_12h_aligned[i] and  # Volume spike
                price > ema_daily_aligned[i]):                    # Price above daily EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: break below support with volume spike and daily bearish trend
            elif (price < support_aligned[i] and 
                  vol_12h_current > 1.8 * vol_ma_12h_aligned[i] and  # Volume spike
                  price < ema_daily_aligned[i]):                    # Price below daily EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below midline
            if price < midline_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above midline
            if price > midline_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Resistance_Support_Volume_DailyTrend"
timeframe = "12h"
leverage = 1.0