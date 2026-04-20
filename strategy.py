#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for long-term trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly EMA(89) for stronger trend
    ema_89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Daily data for entry timing and filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily close for price
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current / 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_89_1w_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_ema_fast = ema_34_1w_aligned[i]
        weekly_ema_slow = ema_89_1w_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio_today = vol_ratio[i]
        
        # Weekly trend alignment: price above both weekly EMAs for uptrend
        weekly_uptrend = (price > weekly_ema_fast) and (weekly_ema_fast > weekly_ema_slow)
        # Price below both weekly EMAs for downtrend
        weekly_downtrend = (price < weekly_ema_fast) and (weekly_ema_fast < weekly_ema_slow)
        
        # Volatility filter: avoid extreme volatility
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr < 2.5 * atr_ma_20
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_today > 1.5)
        
        if position == 0:
            # Enter long in strong weekly uptrend with volume
            if weekly_uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short in strong weekly downtrend with volume
            elif weekly_downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend breakdown or volatility spike
            if (not weekly_uptrend) or (atr > 3.0 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend breakdown or volatility spike
            if (not weekly_downtrend) or (atr > 3.0 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_EMA34_89_Trend_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0