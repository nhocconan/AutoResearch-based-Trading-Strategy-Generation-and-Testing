#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and 1d ATR volatility regime
    # Long when Bull Power > 0 + price > 12h EMA50 + ATR ratio < 0.8 (low volatility)
    # Short when Bear Power < 0 + price < 12h EMA50 + ATR ratio < 0.8 (low volatility)
    # Exit when power crosses zero or ATR ratio > 1.2 (high volatility)
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Elder Ray measures bull/bear strength relative to EMA13
    # 12h EMA50 provides multi-timeframe trend alignment
    # ATR ratio (current/20-period average) filters for low volatility breakouts
    # Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Get 1d data for ATR volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d ATR (14-period) and its 20-period average for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR 14
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[1:15])  # Simple average for first 14 periods
    for i in range(15, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # ATR 20-period average
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio: current ATR / 20-period average ATR
    atr_ratio = atr_14 / np.where(atr_ma_20 == 0, 1, atr_ma_20)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Elder Ray components (Bull Power and Bear Power) on 6h timeframe
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        above_ema_50 = close[i] > ema_50_12h_aligned[i]
        below_ema_50 = close[i] < ema_50_12h_aligned[i]
        
        # Volatility filter: low volatility environment (ATR ratio < 0.8)
        low_volatility = atr_ratio_aligned[i] < 0.8
        
        # High volatility exit condition
        high_volatility = atr_ratio_aligned[i] > 1.2
        
        # Elder Ray signals
        bullish = bull_power[i] > 0
        bearish = bear_power[i] < 0
        
        # Entry conditions
        long_entry = bullish and above_ema_50 and low_volatility
        short_entry = bearish and below_ema_50 and low_volatility
        
        # Exit conditions
        long_exit = not bullish or high_volatility
        short_exit = not bearish or high_volatility
        
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

name = "6h_12h_1d_elder_ray_ema_atr_v1"
timeframe = "6h"
leverage = 1.0