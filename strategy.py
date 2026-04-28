#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA trend filter and ATR-based volume spike filter.
# Enter long when price breaks above 4h Donchian upper channel (20) with 1d EMA50 uptrend and volume > 1.5x ATR-scaled average.
# Enter short when price breaks below 4h Donchian lower channel (20) with 1d EMA50 downtrend and volume confirmation.
# Exit on opposite Donchian channel touch or ATR trailing stop (3x ATR from extreme).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide robust structure, EMA50 on 1d ensures higher-timeframe trend alignment.
# Volume filter uses ATR scaling to adapt to volatility regimes, reducing whipsaws in choppy markets.

name = "4h_Donchian20_Breakout_1dEMA50_ATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:  # Need at least one complete 4h bar for Donchian
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h (shifted by one bar to avoid look-ahead)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (50-period)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR (14) for volatility-based volume filter and stoploss
    # ATR = average of true ranges over 14 periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: >1.5x ATR-scaled 20-bar average volume
    # This adapts volume threshold to current volatility regime
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    atr_scaled_volume_ma = volume_ma_20 * (atr / np.mean(atr[~np.isnan(atr)]) if np.any(~np.isnan(atr)) else 1.0)
    volume_confirm = volume > 1.5 * atr_scaled_volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 50, 14)  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        ema_trend_up = close[i] > ema_50_aligned[i]
        ema_trend_down = close[i] < ema_50_aligned[i]
        
        price = close[i]
        atr_now = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper, price > EMA50 (uptrend), volume confirm
            if price > donchian_high_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_at_entry = atr_now
                highest_since_entry = price
            # Short entry: price < Donchian lower, price < EMA50 (downtrend), volume confirm
            elif price < donchian_low_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_at_entry = atr_now
                lowest_since_entry = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            
            # ATR trailing stop: exit if price drops 3*ATR from highest since entry
            if price < highest_since_entry - 3.0 * atr_now:
                signals[i] = 0.0
                position = 0
            # Exit on Donchian lower touch (reversion to mean)
            elif price <= donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            
            # ATR trailing stop: exit if price rises 3*ATR from lowest since entry
            if price > lowest_since_entry + 3.0 * atr_now:
                signals[i] = 0.0
                position = 0
            # Exit on Donchian upper touch (reversion to mean)
            elif price >= donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals