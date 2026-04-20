#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(13) for long-term trend
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Weekly EMA(34) for trend confirmation
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for entry signals and volatility
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian(20) channels
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily volume data
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_13_1w_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        ema_trend_short = ema_13_1w_aligned[i]
        ema_trend_long = ema_34_1w_aligned[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio_1d = vol_ratio[i]
        
        # Weekly trend alignment: price above both EMAs for uptrend
        trend_up = (price > ema_trend_short) and (ema_trend_short > ema_trend_long)
        # Price below both EMAs for downtrend
        trend_down = (price < ema_trend_short) and (ema_trend_short < ema_trend_long)
        
        # Volatility filter: avoid low volatility (chop) and extreme volatility
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_1d > 1.5)
        
        if position == 0:
            # Enter long on Donchian breakout with weekly uptrend
            if price > upper and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown with weekly downtrend
            elif price < lower and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retrace to midpoint or trend breakdown
            midpoint = (upper + lower) / 2
            if price < midpoint or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retrace to midpoint or trend breakdown
            midpoint = (upper + lower) / 2
            if price > midpoint or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian20_WeeklyTrend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0