#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_RSI50_Trend_Volume
Hypothesis: Combines Donchian channel breakout with RSI(50) trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) with RSI>50 and volume spike.
Short when price breaks below lower Donchian(20) with RSI<50 and volume spike.
Designed for 15-25 trades/year to minimize fee drag while capturing momentum in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # RSI(14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(rsi_values[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from daily EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # RSI filter
        rsi_bullish = rsi_values[i] > 50
        rsi_bearish = rsi_values[i] < 50
        
        # Entry conditions: Donchian breakout with volume, RSI, and trend alignment
        long_entry = (close[i] > high_max[i-1]) and vol_confirm and rsi_bullish and uptrend
        short_entry = (close[i] < low_min[i-1]) and vol_confirm and rsi_bearish and downtrend
        
        # Exit conditions: return to middle of Donchian channel or trend reversal
        donchian_mid = (high_max[i] + low_min[i]) / 2
        long_exit = close[i] < donchian_mid
        short_exit = close[i] > donchian_mid
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_20_Breakout_RSI50_Trend_Volume"
timeframe = "4h"
leverage = 1.0