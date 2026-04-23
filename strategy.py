#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted RSI with 12h Supertrend filter and ATR-based stops
- VW-RSI(14) = RSI calculated on typical price weighted by volume
- Long: VW-RSI < 30 AND price > 12h Supertrend AND ATR(14) < 0.08 * price (low volatility)
- Short: VW-RSI > 70 AND price < 12h Supertrend AND ATR(14) < 0.08 * price
- Exit: VW-RSI crosses 50 in opposite direction
- Uses 12h Supertrend for trend alignment (avoids counter-trend whipsaws)
- Volume-weighted RSI reduces noise from low-volume spikes
- ATR filter ensures entries during consolidation, avoiding false breakouts
- Works in both bull and bear markets by trading mean reversion within the trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Typical price
    tp = (high + low + close) / 3.0
    
    # Volume-weighted RSI calculation
    def vw_rsi(tp, volume, length=14):
        # Calculate changes
        delta = np.diff(tp, prepend=tp[0])
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Volume-weighted gains and losses
        vol_gains = gains * volume
        vol_losses = losses * volume
        
        # Calculate weighted averages
        avg_vg = pd.Series(vol_gains).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_vl = pd.Series(vol_losses).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        
        # Avoid division by zero
        rs = np.where(avg_vl != 0, avg_vg / avg_vl, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    vwrsi = vw_rsi(tp, volume, 14)
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on 12h
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (df_12h['high'] + df_12h['low']) / 2.0
    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(df_12h['close'])
    direction = np.ones_like(df_12h['close'])  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_12h)):
        # Upper Band
        if upper_basic[i] < supertrend[i-1] or df_12h['close'].iloc[i-1] > supertrend[i-1]:
            upper_band = upper_basic[i]
        else:
            upper_band = supertrend[i-1]
            
        # Lower Band
        if lower_basic[i] > supertrend[i-1] or df_12h['close'].iloc[i-1] < supertrend[i-1]:
            lower_band = lower_basic[i]
        else:
            lower_band = supertrend[i-1]
            
        # Supertrend
        if supertrend[i-1] == upper_band:
            if df_12h['close'].iloc[i] > upper_band:
                supertrend[i] = lower_band
                direction[i] = -1
            else:
                supertrend[i] = upper_band
                direction[i] = 1
        else:
            if df_12h['close'].iloc[i] < lower_band:
                supertrend[i] = upper_band
                direction[i] = 1
            else:
                supertrend[i] = lower_band
                direction[i] = -1
    
    # Align HTF Supertrend to LTF
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    
    # ATR for volatility filter (14-period)
    tr_lst = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_lst[0] = high[0] - low[0]
    atr_14 = pd.Series(tr_lst).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # VW-RSI needs 14, Supertrend needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwrsi[i]) or 
            np.isnan(supertrend_aligned[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: ATR < 8% of price (low volatility environment)
        low_vol = atr_14[i] < 0.08 * close[i]
        
        # Trend filter from 12h Supertrend
        uptrend = close[i] > supertrend_aligned[i]
        downtrend = close[i] < supertrend_aligned[i]
        
        if position == 0:
            # Long: VW-RSI oversold + uptrend + low volatility
            if vwrsi[i] < 30 and uptrend and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: VW-RSI overbought + downtrend + low volatility
            elif vwrsi[i] > 70 and downtrend and low_vol:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: VW-RSI crosses 50 in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: VW-RSI crosses above 50
                if vwrsi[i] > 50:
                    exit_signal = True
            elif position == -1:
                # Exit short: VW-RSI crosses below 50
                if vwrsi[i] < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_VolumeWeightedRSI_12hSupertrend_ATRFilter"
timeframe = "6h"
leverage = 1.0