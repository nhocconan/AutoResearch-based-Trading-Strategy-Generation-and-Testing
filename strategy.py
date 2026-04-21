#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation.
# Long when price breaks above upper Donchian in uptrend (1w EMA > price), short when breaks below lower Donchian in downtrend (1w EMA < price).
# Volume > 1.5x 20-period average confirms breakout strength. EMA filter avoids whipsaws in ranging markets.
# Target: 10-25 trades/year by requiring strong trend + volume + breakout alignment.
# Works in bull/bear: EMA filter ensures only aligned trends are traded, avoiding counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).values
    
    # Align EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 20-period Donchian channels on 1d data
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    upper = high_roll.values
    lower = low_roll.values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1w EMA
        price_above_ema = price > ema_1w_aligned[i]
        price_below_ema = price < ema_1w_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above upper Donchian and above 1w EMA
                if price > upper[i] and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian and below 1w EMA
                elif price < lower[i] and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below lower Donchian or crosses below 1w EMA
                if price < lower[i] or price < ema_1w_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above upper Donchian or crosses above 1w EMA
                if price > upper[i] or price > ema_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0