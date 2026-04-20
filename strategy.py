#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Keltner_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # === Weekly: Keltner Channel (20 EMA + 2*ATR) for trend filter ===
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # EMA20 on weekly close
    ema20_weekly = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR(10) for weekly
    tr1 = np.abs(weekly_high[1:] - weekly_low[1:])
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
    tr_weekly = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_weekly = np.concatenate([[np.nan], tr_weekly])
    atr_weekly = pd.Series(tr_weekly).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel bounds
    kc_upper_weekly = ema20_weekly + 2.0 * atr_weekly
    kc_lower_weekly = ema20_weekly - 2.0 * atr_weekly
    
    # Align weekly Keltner bounds to daily
    kc_upper_aligned = align_htf_to_ltf(prices, df_weekly, kc_upper_weekly)
    kc_lower_aligned = align_htf_to_ltf(prices, df_weekly, kc_lower_weekly)
    ema20_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # === Daily Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stop loss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Get values
        kc_upper = kc_upper_aligned[i]
        kc_lower = kc_lower_aligned[i]
        ema20 = ema20_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        current_atr = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(kc_upper) or np.isnan(kc_lower) or np.isnan(ema20) or 
            np.isnan(current_vol_ma) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-day average
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        if position == 0:
            # Long: close > weekly Keltner upper + volume breakout
            if current_close > kc_upper and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: close < weekly Keltner lower + volume breakout
            elif current_close < kc_lower and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: close < weekly EMA20 OR stop loss
            if current_close < ema20 or current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close > weekly EMA20 OR stop loss
            if current_close > ema20 or current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals