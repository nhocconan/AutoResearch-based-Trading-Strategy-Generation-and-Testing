#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and ATR-based volatility filter
# Long when price breaks above 20-day Donchian high AND price > 1w EMA200 AND ATR(14) > 0.5 * ATR(50)
# Short when price breaks below 20-day Donchian low AND price < 1w EMA200 AND ATR(14) > 0.5 * ATR(50)
# Exit when price retests the midpoint of the Donchian channel
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years) to avoid overtrading.
# Focuses on high-probability breakouts in strong trends while filtering low-volatility environments.
# Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns
# with 1w trend filter preventing counter-trend trades and volatility filter ensuring sufficient momentum.

name = "1d_Donchian20_VolatilityFilter_1wEMA200_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    # True Range components
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility filter: ATR(14) > 0.5 * ATR(50) (ensures sufficient momentum)
    vol_filter = atr_14 > (0.5 * atr_50)
    
    # Calculate Donchian(20) channels
    # Highest high of last 20 periods (including current)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods (including current)
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Midpoint of Donchian channel for exit
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 200)  # Donchian(20), ATR(50), and EMA200_1w warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = vol_filter[i]
        curr_ema200_1w = ema_200_1w_aligned[i]
        curr_highest = highest_20[i]
        curr_lowest = lowest_20[i]
        curr_mid = donchian_mid[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian midpoint
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian midpoint
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above 20-day Donchian high AND price > 1w EMA200 AND volatility filter
            if curr_close > curr_highest and curr_close > curr_ema200_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-day Donchian low AND price < 1w EMA200 AND volatility filter
            elif curr_close < curr_lowest and curr_close < curr_ema200_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals