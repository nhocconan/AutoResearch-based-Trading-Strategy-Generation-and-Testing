#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Long when price breaks above 20-bar Donchian high AND price > 1d EMA50 AND volume > 1.5x 20-bar average
# Short when price breaks below 20-bar Donchian low AND price < 1d EMA50 AND volume > 1.5x 20-bar average
# Exit when price crosses the 20-bar Donchian midpoint OR ATR-based stoploss (2.0 * ATR)
# Uses 12h timeframe targeting 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Works in bull markets via Donchian breakouts with uptrend filter and in bear markets via breakdowns with downtrend filter.

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    # ATR for stoploss (14-period)
    tr1 = high_series[1:] - low_series[:-1]
    tr2 = np.abs(high_series[1:] - close_series[:-1])
    tr3 = np.abs(low_series[1:] - close_series[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Donchian(20) needs 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        atr_val = atr[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > upper Donchian AND price > 1d EMA50 AND volume spike
            if price > upper and price > ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price < lower Donchian AND price < 1d EMA50 AND volume spike
            elif price < lower and price < ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss, midpoint cross, or breakdown
            # ATR-based stoploss: 2.0 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss, price < midpoint, or price < lower Donchian (breakdown)
            if price < stop_loss or price < mid or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss, midpoint cross, or breakout
            # ATR-based stoploss: 2.0 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss, price > midpoint, or price > upper Donchian (breakout)
            if price > stop_loss or price > mid or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals