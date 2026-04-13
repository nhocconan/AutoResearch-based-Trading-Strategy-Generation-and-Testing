#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
    # Long: price > upper Donchian(20) AND price > 1w EMA50 AND volume > 1.5x avg
    # Short: price < lower Donchian(20) AND price < 1w EMA50 AND volume > 1.5x avg
    # Exit: opposite Donchian breakout or volume dry-up
    # Using 12h timeframe for low trade frequency, Donchian for structure,
    # 1w EMA50 for regime filter (avoid counter-trend trades), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h Donchian channels (20-period)
    upper_donch = np.full(n, np.nan)
    lower_donch = np.full(n, np.nan)
    for i in range(20, n):
        upper_donch[i] = np.max(high[i-20:i])
        lower_donch[i] = np.min(low[i-20:i])
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: price > weekly EMA50 = bullish bias, price < weekly EMA50 = bearish bias
        bullish_bias = close[i] > ema_1w_aligned[i]
        bearish_bias = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + regime bias + volume confirmation
        long_entry = (close[i] > upper_donch[i]) and bullish_bias and vol_confirm
        short_entry = (close[i] < lower_donch[i]) and bearish_bias and vol_confirm
        
        # Exit logic: opposite Donchian breakout or volume dry-up
        long_exit = (close[i] < lower_donch[i]) or not vol_confirm
        short_exit = (close[i] > upper_donch[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_donchian_ema_volume_v1"
timeframe = "12h"
leverage = 1.0