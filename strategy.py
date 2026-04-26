#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation. 
In bull markets (price > 12h EMA50), long on break above R3; in bear markets (price < 12h EMA50), short on break below S3.
Volume spike (>1.5x 20-period average) confirms breakout strength. Uses discrete sizing 0.25 to limit trades (~20-40/year).
Works in bull/bear via 12h trend filter and volatility-based stoploss (2*ATR).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate previous day's OHLC for Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to LTF
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # ATR for volatility filtering and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volume spike confirmation: >1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 12h EMA, 2 for 1d OHLC, 20 for volume MA, 14 for ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > 12h EMA50 (bull trend) AND breaks above R3 AND volume spike
            long_entry = (close_val > ema_50_val) and (close_val > r3_val) and vol_spike
            # Short: price < 12h EMA50 (bear trend) AND breaks below S3 AND volume spike
            short_entry = (close_val < ema_50_val) and (close_val < s3_val) and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or ATR-based stop
            # Exit if price drops below 12h EMA50 OR 2*ATR from entry
            if close_val < ema_50_val or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or ATR-based stop
            # Exit if price rises above 12h EMA50 OR 2*ATR from entry
            if close_val > ema_50_val or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0