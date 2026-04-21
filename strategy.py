#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1-day volume surge and trend filter.
Long when price breaks above 20-period high with volume > 2x 20-period average and close > EMA(50);
Short when price breaks below 20-period low with volume > 2x average and close < EMA(50).
Exit on opposite Donchian break or 2x ATR stop. Designed for 20-40 trades/year to minimize fee drag.
Works in bull markets via breakouts and in bear via short breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Donchian channels (20-period) on 4h data
    high_20 = pd.Series(prices['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values
    
    # EMA(50) for trend filter on 4h close
    ema_50 = pd.Series(prices['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for stop (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current 1-day volume aligned to 4h
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_1d_current = vol_1d_aligned[i]
        
        if position == 0:
            # Enter long: break above 20-period high with volume surge and close > EMA50
            if (price_high > high_20[i] and 
                vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i] and
                price_close > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: break below 20-period low with volume surge and close < EMA50
            elif (price_low < low_20[i] and 
                  vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i] and
                  price_close < ema_50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite Donchian break or 2x ATR stop
            exit_signal = False
            
            if position == 1:
                # Exit long: break below 20-period low OR price < entry - 2*ATR
                if price_low < low_20[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use high_20 as entry level for long
                    entry_level = high_20[i-20] if i >= 20 else high_20[0]
                    if price_close < entry_level - 2.0 * atr[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: break above 20-period high OR price > entry + 2*ATR
                if price_high > high_20[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use low_20 as entry level for short
                    entry_level = low_20[i-20] if i >= 20 else low_20[0]
                    if price_close > entry_level + 2.0 * atr[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_VolumeSurge2x_EMA50Trend_ATR2x"
timeframe = "4h"
leverage = 1.0