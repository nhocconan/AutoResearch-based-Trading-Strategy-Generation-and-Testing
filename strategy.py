#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ADX (14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR (14) for volatility filter and stoploss
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR (14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = prices['high'].iloc[i]
        low_val = prices['low'].iloc[i]
        adx_val = adx_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_12h_val = atr_12h[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(atr_1d_val) or 
            np.isnan(atr_12h_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions: strong trend (ADX > 25) and low volatility (ATR < 1.5 * median ATR)
        if position == 0:
            # Calculate median ATR for volatility filter
            if i >= 50:
                atr_median = np.median(atr_12h[max(0, i-50):i])
                vol_filter = atr_12h_val < 1.5 * atr_median if not np.isnan(atr_median) else True
            else:
                vol_filter = True
            
            if adx_val > 25 and vol_filter:
                # Long: bullish engulfing pattern
                if (close_val > prices['open'].iloc[i] and 
                    prices['close'].iloc[i-1] < prices['open'].iloc[i-1] and
                    close_val > prices['close'].iloc[i-1] and
                    prices['open'].iloc[i] < prices['close'].iloc[i-1]):
                    signals[i] = 0.25
                    position = 1
                # Short: bearish engulfing pattern
                elif (close_val < prices['open'].iloc[i] and 
                      prices['close'].iloc[i-1] > prices['open'].iloc[i-1] and
                      close_val < prices['close'].iloc[i-1] and
                      prices['open'].iloc[i] > prices['close'].iloc[i-1]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: bearish engulfing or ATR-based stop
            bearish_engulf = (close_val < prices['open'].iloc[i] and 
                             prices['close'].iloc[i-1] > prices['open'].iloc[i-1] and
                             close_val < prices['close'].iloc[i-1] and
                             prices['open'].iloc[i] > prices['close'].iloc[i-1])
            if bearish_engulf or close_val < high_val - 2.0 * atr_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish engulfing or ATR-based stop
            bullish_engulf = (close_val > prices['open'].iloc[i] and 
                             prices['close'].iloc[i-1] < prices['open'].iloc[i-1] and
                             close_val > prices['close'].iloc[i-1] and
                             prices['open'].iloc[i] < prices['close'].iloc[i-1])
            if bullish_engulf or close_val > low_val + 2.0 * atr_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_ADX_Engulfing_ATRFilter_V1
# Uses 1d ADX for trend strength and 12h engulfing patterns for entry
# Enters long on bullish engulfing when ADX > 25 and volatility is low
# Enters short on bearish engulfing when ADX > 25 and volatility is low
# Uses ATR-based stop loss (2 * ATR) for risk management
# Designed for 12h timeframe with ~12-37 trades/year
name = "12h_ADX_Engulfing_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0