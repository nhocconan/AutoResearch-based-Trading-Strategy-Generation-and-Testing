#!/usr/bin/env python3
# 1d_weekly_keltner_breakout_volume_v1
# Hypothesis: Daily timeframe strategy using weekly Keltner channel breakouts with volume confirmation.
# Works in both bull and bear markets by trading volatility expansions in the direction of the weekly trend.
# Long: price breaks above weekly Keltner Upper Band (EMA20 + 2*ATR) with volume > 1.5x 20-day average
# Short: price breaks below weekly Keltner Lower Band (EMA20 - 2*ATR) with volume > 1.5x 20-day average
# Exit: price returns to weekly EMA20 middle band or ATR-based stoploss (2.0x ATR)
# Uses 1d primary timeframe with 1w HTF for Keltner calculation.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_keltner_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for stoploss and Keltner channels
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.mean(tr[i-20:i])
    
    # Calculate volume ratio (current vs 20-day average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1w data for Keltner channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for Keltner middle band
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR(20) for Keltner width
    tr_1w = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        atr_1w[i] = np.mean(tr_1w[i-20:i])
    
    # Calculate weekly Keltner channels
    keltner_mid_1w = ema_20_1w
    keltner_upper_1w = keltner_mid_1w + 2.0 * atr_1w
    keltner_lower_1w = keltner_mid_1w - 2.0 * atr_1w
    
    # Align 1w Keltner levels to daily timeframe
    keltner_mid_aligned = align_htf_to_ltf(prices, df_1w, keltner_mid_1w)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper_1w)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(keltner_mid_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle band OR stoploss hit (2.0x ATR below entry)
            if price <= keltner_mid_aligned[i] or price <= entry_price - 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle band OR stoploss hit (2.0x ATR above entry)
            if price >= keltner_mid_aligned[i] or price >= entry_price + 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above upper band with volume confirmation
            if price > keltner_upper_aligned[i] and vol_r > 1.5:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume confirmation
            elif price < keltner_lower_aligned[i] and vol_r > 1.5:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals