#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume spike confirmation
    # Long: price breaks above Donchian(20) high AND price > 1d EMA200 AND volume > 1.5x avg
    # Short: price breaks below Donchian(20) low AND price < 1d EMA200 AND volume > 1.5x avg
    # Exit: opposite Donchian breakout or volume dry-up
    # Using 4h primary timeframe for balance of trade frequency and signal quality,
    # Donchian for objective breakout levels, 1d EMA200 for trend filter, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate Donchian(20) on 4h data
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate volume spike (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1d_200_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA200 = bullish bias, price < EMA200 = bearish bias
        bullish_bias = close[i] > ema_1d_200_aligned[i]
        bearish_bias = close[i] < ema_1d_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + trend bias + volume confirmation
        long_entry = (close[i] > donchian_high[i]) and bullish_bias and vol_confirm
        short_entry = (close[i] < donchian_low[i]) and bearish_bias and vol_confirm
        
        # Exit logic: opposite Donchian breakout or volume dry-up
        long_exit = (close[i] < donchian_low[i]) or not vol_confirm
        short_exit = (close[i] > donchian_high[i]) or not vol_confirm
        
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

name = "4h_1d_donchian_breakout_ema200_volume_v1"
timeframe = "4h"
leverage = 1.0