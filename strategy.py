#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 12-hour Donchian breakout with 1-day volume confirmation and 1-day ATR filter
# Long when price breaks above 12-hour Donchian upper channel (20-period high) with 1-day volume > 1.5x 1-day average and 1-day ATR > 0.5% of price
# Short when price breaks below 12-hour Donchian lower channel (20-period low) with same conditions
# Uses 12-hour Donchian for entry timing, 1-day volume/ATR for regime filtering to avoid chop
# Designed to work in trending markets (bull/bear) with volume confirmation and avoid false breakouts in low volatility
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_12hDonchian20_1dVolATR_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12-hour Donchian Channel (20-period high/low) on 12h timeframe
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-day ATR (14-period) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range for 1d
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # 1-day average volume (20-period)
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align 1-day indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Price-based volatility filter: ATR > 0.5% of price
    price_for_vol = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    vol_filter = atr_14_aligned > (0.005 * price_for_vol)
    
    # Volume confirmation: current 1-day volume > 1.5x 20-day average
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
    volume_filter = vol_1d_aligned > (1.5 * vol_ma_20_aligned)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 12h Donchian upper with vol/vol filter
            if close[i] > high_20[i] and vol_filter[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 12h Donchian lower with vol/vol filter
            elif close[i] < low_20[i] and vol_filter[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 12h Donchian lower (support break)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 12h Donchian upper (resistance break)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals