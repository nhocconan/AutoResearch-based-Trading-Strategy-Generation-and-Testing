#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter and volume confirmation.
# Uses Williams %R(14) on 6h for overbought/oversold signals (> -20 = overbought, < -80 = oversold).
# Weekly trend filter: price above/below weekly EMA50 determines bias.
# In uptrend (price > weekly EMA50): only take longs from oversold.
# In downtrend (price < weekly EMA50): only take shorts from overbought.
# Volume confirmation (>1.5x average) required for entries.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6d data (Williams %R calculation window) ===
    df_6d = get_htf_data(prices, '6d')
    high_6d = df_6d['high'].values
    low_6d = df_6d['low'].values
    close_6d = df_6d['close'].values
    volume_6d = df_6d['volume'].values
    
    # === 1d data (for weekly aggregation) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === Williams %R(14) on 6d data ===
    highest_high = pd.Series(high_6d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6d).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close_6d) / (highest_high - lowest_low)
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)  # avoid division by zero
    willr_6d_aligned = align_htf_to_ltf(prices, df_6d, willr)
    
    # === Weekly EMA50 from daily data ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Aggregate to weekly: take last value of each week
    weekly_ema50 = []
    week_start = 0
    for i in range(len(close_1d)):
        if i == 0 or pd.Timestamp(close_1d[i]).isocalendar().week != pd.Timestamp(close_1d[i-1]).isocalendar().week or i == len(close_1d)-1:
            if i > week_start:
                weekly_ema50.append(ema50_1d[i-1])
            week_start = i
    if len(close_1d) > week_start:
        weekly_ema50.append(ema50_1d[-1])
    weekly_ema50 = np.array(weekly_ema50)
    
    # Create weekly DataFrame for alignment
    weekly_times = []
    current_week_start = 0
    for i in range(len(close_1d)):
        if i == 0 or (i > 0 and pd.Timestamp(close_1d[i]).isocalendar().week != pd.Timestamp(close_1d[i-1]).isocalendar().week):
            weekly_times.append(pd.Timestamp(close_1d[i-1]))
            current_week_start = i
    if len(close_1d) > current_week_start:
        weekly_times.append(pd.Timestamp(close_1d[-1]))
    
    df_weekly = pd.DataFrame({'close': weekly_ema50}, index=pd.DatetimeIndex(weekly_times))
    df_weekly_6h = get_htf_data(prices, '6h')  # dummy to get index
    # Reindex weekly EMA50 to 6h index using forward fill
    weekly_ema50_series = pd.Series(weekly_ema50, index=df_weekly.index)
    weekly_ema50_6h = weekly_ema50_series.reindex(df_weekly_6h.index, method='ffill').values
    
    # === 6d volume ratio for confirmation ===
    vol_ma_10_6d = pd.Series(volume_6d).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6d = volume_6d / vol_ma_10_6d
    vol_ratio_6d_aligned = align_htf_to_ltf(prices, df_6d, vol_ratio_6d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(willr_6d_aligned[i]) or 
            np.isnan(weekly_ema50_6h[i]) or
            np.isnan(vol_ratio_6d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        willr_val = willr_6d_aligned[i]
        weekly_ema50_val = weekly_ema50_6h[i]
        vol_ratio = vol_ratio_6d_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.5 * ATR(14)
            # Simplified: use 2.5% of price as proxy for ATR
            if price < entry_price * 0.975:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.5 * ATR(14)
            if price > entry_price * 1.025:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Williams %R reaches overbought or trend changes
            if willr_val >= -20 or price < weekly_ema50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R reaches oversold or trend changes
            if willr_val <= -80 or price > weekly_ema50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Uptrend: price above weekly EMA50 -> look for longs from oversold
            if price > weekly_ema50_val:
                if willr_val <= -80 and vol_ratio > 1.5:  # Oversold with volume
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
            # Downtrend: price below weekly EMA50 -> look for shorts from overbought
            elif price < weekly_ema50_val:
                if willr_val >= -20 and vol_ratio > 1.5:  # Overbought with volume
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_WeeklyTrend_Filter_Volume_v1"
timeframe = "6h"
leverage = 1.0