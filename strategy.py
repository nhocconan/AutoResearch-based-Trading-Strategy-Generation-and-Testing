#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R1/S1 breakout with 1d ATR volatility filter and volume spike confirmation.
Long when price breaks above Camarilla R1 level with ATR(14) > 1.5x its 20-period MA (high volatility regime) and volume > 2.0x average.
Short when price breaks below Camarilla S1 level with same volatility and volume conditions.
Exit on opposite Camarilla level break or when volatility drops below ATR(14) < 1.0x its 20-period MA (low volatility regime).
Uses 6h timeframe targeting 50-150 total trades over 4 years. Camarilla R1/S1 levels provide intraday support/resistance.
ATR filter ensures we only trade during volatile markets where breakouts are meaningful, avoiding choppy regimes.
Volume spike confirms breakout institutional participation. Designed to work in both bull (breakouts continuation) and bear (breakdown continuation) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr14).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and its MA to 6h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr14_aligned[i]) or np.isnan(atr_ma_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_val = atr14_aligned[i]
        atr_ma_val = atr_ma_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Calculate Camarilla levels for today using previous day's OHLC
        if i >= 2:
            lookback_start = i - 2
            prev_high = np.max(high[lookback_start:i])
            prev_low = np.min(low[lookback_start:i])
            prev_close = close[i-1]  # previous bar close
            
            # Camarilla levels
            range_val = prev_high - prev_low
            camarilla_r1 = prev_close + (range_val * 1.1 / 12)
            camarilla_s1 = prev_close - (range_val * 1.1 / 12)
            camarilla_r4 = prev_close + (range_val * 1.1 / 2)
            camarilla_s4 = prev_close - (range_val * 1.1 / 2)
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND high volatility (ATR > MA) AND volume spike
            if (price > camarilla_r1 and atr_val > atr_ma_val * 1.5 and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S1 AND high volatility (ATR > MA) AND volume spike
            elif (price < camarilla_s1 and atr_val > atr_ma_val * 1.5 and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S1 OR volatility drops (ATR < MA)
                if (price < camarilla_s1 or atr_val < atr_ma_val * 1.0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R1 OR volatility drops (ATR < MA)
                if (price > camarilla_r1 or atr_val < atr_ma_val * 1.0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R1_S1_1dATR_VolumeSpike"
timeframe = "6h"
leverage = 1.0