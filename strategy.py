#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d trend filter
# - Long when price breaks above 4h Donchian upper channel (20-period) with 1h volume spike and 1d uptrend (close > EMA50)
# - Short when price breaks below 4h Donchian lower channel (20-period) with 1h volume spike and 1d downtrend (close < EMA50)
# - Uses discrete position sizing (0.20) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or price reverts to 4h Donchian midpoint
# - Session filter: 08-20 UTC to reduce noise trades
# - Targets 15-35 trades/year (60-140 total over 4 years) to avoid fee drag

name = "1h_4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours ONCE
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    donchian_len = 20
    upper_4h = np.full_like(high_4h, np.nan)
    lower_4h = np.full_like(low_4h, np.nan)
    for i in range(donchian_len - 1, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-donchian_len+1:i+1])
        lower_4h[i] = np.min(low_4h[i-donchian_len+1:i+1])
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    midpoint_4h = (upper_4h + lower_4h) / 2.0
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, midpoint_4h)
    
    # 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_4h = np.zeros_like(tr)
    atr_14_4h[donchian_len-1] = np.mean(tr[:donchian_len])
    for i in range(donchian_len, len(tr)):
        atr_14_4h[i] = (atr_14_4h[i-1] * (donchian_len-1) + tr[i]) / donchian_len
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(prices['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'].values > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(midpoint_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to 4h Donchian midpoint (mean reversion)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] > midpoint_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to 4h Donchian midpoint (mean reversion)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] < midpoint_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for 4h Donchian breakout with volume and trend filters
            if vol_spike[i]:
                # Long signal: price breaks above 4h upper channel in 1d uptrend
                if (prices['high'].iloc[i] > upper_4h_aligned[i] and 
                    prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h_aligned[i]
                    signals[i] = 0.20
                # Short signal: price breaks below 4h lower channel in 1d downtrend
                elif (prices['low'].iloc[i] < lower_4h_aligned[i] and 
                      prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h_aligned[i]
                    signals[i] = -0.20
    
    return signals