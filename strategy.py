#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for trend and volatility context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 1d ATR percentile (20-period) for volatility regime
    atr_percentile = pd.Series(atr_14).rolling(window=20, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) == 20 else np.nan, raw=True
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1w ATR(14) for weekly volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1_w = np.abs(high_1w - low_1w)
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w ATR percentile (20-period) for weekly volatility regime
    atr_percentile_w = pd.Series(atr_14_w).rolling(window=20, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) == 20 else np.nan, raw=True
    ).values
    
    # Align weekly ATR percentile to 12h
    atr_percentile_w_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile_w)
    
    # Calculate 12h price change for momentum
    price_change = (prices['close'].values / np.roll(prices['close'].values, 12) - 1)  # 12-period (1 day) return
    price_change[0:12] = 0  # First 12 bars: no data
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or
            np.isnan(atr_percentile_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        atr_val = atr_14_aligned[i]
        atr_pct = atr_percentile_aligned[i]
        atr_pct_w = atr_percentile_w_aligned[i]
        mom = price_change[i]
        
        # Volatility filter: only trade when volatility is above median (expanding)
        vol_filter = atr_pct > atr_pct_w  # Daily volatility > weekly median volatility
        
        # Momentum filter: require significant price movement
        mom_filter = np.abs(mom) > 0.02  # 2% minimum daily move
        
        if position == 0:
            # Enter long: bullish momentum + volatility expansion
            if (mom > 0 and vol_filter and mom_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish momentum + volatility expansion
            elif (mom < 0 and vol_filter and mom_filter):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: momentum reversal or volatility contraction
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish momentum OR volatility contraction
                if (mom < -0.01) or (atr_pct < atr_pct_w * 0.8):  # Strong reversal or vol contraction
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish momentum OR volatility contraction
                if (mom > 0.01) or (atr_pct < atr_pct_w * 0.8):  # Strong reversal or vol contraction
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_VolMom_Expansion"
timeframe = "12h"
leverage = 1.0