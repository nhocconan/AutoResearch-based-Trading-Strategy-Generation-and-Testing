#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# Long when: price breaks above Donchian(20) high, price > 12h EMA(50), volume > 1.5x 20-bar average
# Short when: price breaks below Donchian(20) low, price < 12h EMA(50), volume > 1.5x 20-bar average
# Exit when: price crosses opposite Donchian(20) boundary OR trailing stop at 2*ATR(14)
# Target: 75-200 trades over 4 years on 4h timeframe

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR(14) for stop loss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit conditions
            exit_condition = False
            if close[i] <= donchian_low[i]:  # reversed breakout
                exit_condition = True
            elif close[i] <= entry_price - 2.0 * atr[i]:  # stop loss
                exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # short position
            # Exit conditions
            exit_condition = False
            if close[i] >= donchian_high[i]:  # reversed breakout
                exit_condition = True
            elif close[i] >= entry_price + 2.0 * atr[i]:  # stop loss
                exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and close[i] > ema_50_12h_aligned[i]:
                    # Bullish breakout above 12h EMA
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < donchian_low[i] and close[i] < ema_50_12h_aligned[i]:
                    # Bearish breakout below 12h EMA
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals