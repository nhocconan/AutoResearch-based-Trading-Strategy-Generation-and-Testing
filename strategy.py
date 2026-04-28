#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Pullback_Volume_Regime
Hypothesis: Uses 12h KAMA for primary trend direction (10-period ER), with RSI(14) pullback entries during low volatility (Choppiness Index > 61.8) and volume confirmation (1.5x 24-bar average). Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets by following KAMA direction and using pullbacks in ranging conditions to avoid whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Choppiness Index regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    atr_1w = []
    for i in range(len(close_1w)):
        if i == 0:
            tr = high_1w[0] - low_1w[0]
        else:
            tr = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        atr_1w.append(tr)
    
    atr_1w = np.array(atr_1w)
    atr_sum_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    high_low_range_1w = pd.Series(high_1w - low_1w).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum_1w / high_low_range_1w) / np.log10(14)
    chop = np.where(high_low_range_1w == 0, 100, chop)
    chop_align = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Get 12h data for KAMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h KAMA (10-period ER)
    close_12h = df_12h['close'].values
    change_12h = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility_12h = np.abs(np.diff(close_12h, 1))
    
    er_12h = np.where(volatility_12h > 0, change_12h / volatility_12h, 0)
    sc_12h = (er_12h * (0.6645 - 0.0645) + 0.0645) ** 2
    
    kama_12h = np.full_like(close_12h, np.nan)
    kama_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    
    kama_12h_align = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Volume confirmation: >1.5x 24-period MA (2 days of 12h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_align[i]) or 
            np.isnan(chop_align[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h KAMA
        uptrend = close[i] > kama_12h_align[i]
        downtrend = close[i] < kama_12h_align[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_24[i])
        
        # Range filter: Choppiness Index > 61.8 (ranging market)
        range_filter = chop_align[i] > 61.8
        
        # RSI(14) for pullback entries
        if i >= 14:
            rsi_period = 14
            delta = np.diff(close[max(0, i-rsi_period):i+1])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-rsi_period:]) if len(gain) >= rsi_period else 0
            avg_loss = np.mean(loss[-rsi_period:]) if len(loss) >= rsi_period else 0
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 50
        else:
            rsi = 50
        
        # RSI pullback conditions: RSI < 40 in uptrend, RSI > 60 in downtrend
        rsi_long = rsi < 40
        rsi_short = rsi > 60
        
        # Entry conditions
        long_entry = uptrend and vol_confirm and range_filter and rsi_long
        short_entry = downtrend and vol_confirm and range_filter and rsi_short
        
        # Exit conditions: reverse signal or loss of trend/volume
        long_exit = not uptrend or not vol_confirm or not range_filter
        short_exit = not downtrend or not vol_confirm or not range_filter
        
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

name = "12h_KAMA_Direction_RSI_Pullback_Volume_Regime"
timeframe = "12h"
leverage = 1.0