#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band AND close > 1d EMA50 AND volume > 2x 20-period MA.
# Short when price breaks below Donchian lower band AND close < 1d EMA50 AND volume > 2x 20-period MA.
# Uses ATR(14) stoploss: exit long when price < highest_high - 2.5*ATR, exit short when price < lowest_low + 2.5*ATR.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) with discrete sizing 0.25.
# Works in bull via breakout longs and bear via breakdown shorts when aligned with 1d trend.

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATR"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper AND above 1d EMA50 AND volume spike
            if close_val > highest_high[i] and close_val > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            # Short entry: price breaks below Donchian lower AND below 1d EMA50 AND volume spike
            elif close_val < lowest_low[i] and close_val < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high_val)
            # Long exit: price drops below highest_high - 2.5*ATR OR Donchian break down OR volume drops
            if (close_val < highest_since_entry - 2.5 * atr_val or 
                close_val < lowest_low[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Short exit: price rises above lowest_low + 2.5*ATR OR Donchian break up OR volume drops
            if (close_val > lowest_since_entry + 2.5 * atr_val or 
                close_val > highest_high[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals