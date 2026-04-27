#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
# Works in bull markets by capturing breakouts, in bear markets by avoiding counter-trend trades via weekly filter.
# Uses 1d as primary timeframe, 1w for trend filter. Targets 20-50 trades over 4 years (~5-12/year) to minimize fee drag.
# Entry: Price breaks Donchian(20) high/low with volume > 2x 20-day average and price vs weekly EMA34 alignment.
# Exit: Opposite Donchian break or 2x ATR trailing stop. Position size 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_34 = np.full(len(close_1w), np.nan)
    if len(close_1w) > 0:
        ema_34[0] = close_1w[0]
        alpha = 2 / (34 + 1)
        for i in range(1, len(close_1w)):
            ema_34[i] = alpha * close_1w[i] + (1 - alpha) * ema_34[i-1]
    
    # Align weekly EMA to daily
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate 22-period ATR (approx 1 month) for volatility and stop
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(len(tr), np.nan)
    atr_period = 22
    for i in range(atr_period, len(tr)):
        if i == atr_period:
            atr[i] = np.mean(tr[1:atr_period+1])
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 20-period high/low for Donchian breakout
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 20
    for i in range(period, n):
        high_max[i] = np.max(high[i-period:i])
        low_min[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(atr_period, vol_period, period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and price above weekly EMA34
            if price > high_max[i] and vol_ratio > 2.0 and price > ema_34_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume and price below weekly EMA34
            elif price < low_min[i] and vol_ratio > 2.0 and price < ema_34_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 2x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 2x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_EMA34_Trend_Volume_ATRStop_v1"
timeframe = "1d"
leverage = 1.0