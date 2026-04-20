# 6h_12h_Trend_Filtered_Price_Action
# Trend following with 12h EMA filter and 6h price action
# Works in bull/bear: trend filter prevents counter-trend trades
# 12h EMA200 defines regime, 6h price action (high/low vs prior) triggers entries
# Volume filter ensures conviction
# Target: 20-40 trades/year per symbol

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Trend_Filtered_Price_Action"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h: EMA200 trend filter ===
    close_12h = df_12h['close'].values
    ema200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_12h, ema200)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 50-period average)
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.where(vol_ma50 > 0, vol_ma50, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price action signals
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        if position == 0:
            # Long: price makes higher high AND higher low in uptrend
            if (high_val > prev_high and low_val > prev_low and  # HH HL
                close_val > ema_val and                          # Price above EMA200
                vol_ratio_val > 1.3):                            # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: price makes lower high AND lower low in downtrend
            elif (high_val < prev_high and low_val < prev_low and  # LH LL
                  close_val < ema_val and                          # Price below EMA200
                  vol_ratio_val > 1.3):                            # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or loss of momentum
            if (close_val < ema_val or                    # Price below EMA200
                (high_val < prev_high and low_val < prev_low)):  # LH LL
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or loss of momentum
            if (close_val > ema_val or                    # Price above EMA200
                (high_val > prev_high and low_val > prev_low)):  # HH HL
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals