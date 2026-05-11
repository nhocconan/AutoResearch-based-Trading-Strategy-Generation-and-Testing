#!/usr/bin/env python3
name = "6h_HeikinAshi_Trend_12hTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Heikin-Ashi calculation
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Heikin-Ashi close
    ha_close = (open_ + high + low + close) / 4
    
    # Calculate Heikin-Ashi open
    ha_open = np.zeros_like(open_)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    
    # Calculate Heikin-Ashi high and low
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    
    # Trend detection: consecutive same-color HA candles
    ha_bullish = ha_close > ha_open
    ha_bearish = ha_close < ha_open
    
    # Count consecutive bullish/bearish candles
    consec_bull = np.zeros(n, dtype=int)
    consec_bear = np.zeros(n, dtype=int)
    
    for i in range(1, n):
        if ha_bullish[i]:
            consec_bull[i] = consec_bull[i-1] + 1
            consec_bear[i] = 0
        elif ha_bearish[i]:
            consec_bear[i] = consec_bear[i-1] + 1
            consec_bull[i] = 0
        else:
            consec_bull[i] = 0
            consec_bear[i] = 0
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: 3+ consecutive bullish HA candles + price above 12h EMA50 + volume surge
            if (consec_bull[i] >= 3 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: 3+ consecutive bearish HA candles + price below 12h EMA50 + volume surge
            elif (consec_bear[i] >= 3 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or volume drops
            if position == 1:
                # Exit long: bearish reversal or price below EMA
                if (consec_bear[i] >= 2) or (close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish reversal or price above EMA
                if (consec_bull[i] >= 2) or (close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals