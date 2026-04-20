#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for long-term trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # 1h price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h volume filter (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    # Hour filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend_long = ema_50_1d_aligned[i]
        adx = adx_14_1d_aligned[i]
        vol_ratio_1h = vol_ratio[i]
        
        # Multi-timeframe trend filter: price above daily EMA50 for uptrend
        trend_up = price > ema_trend_long
        # Price below daily EMA50 for downtrend
        trend_down = price < ema_trend_long
        
        # Trend strength filter: require strong trend (ADX > 25)
        trend_filter = adx > 25
        
        # Volume filter: require above-average volume
        vol_filter = vol_ratio_1h > 1.5
        
        if position == 0:
            # Enter long in strong uptrend with volume and trend strength
            if trend_up and trend_filter and vol_filter:
                signals[i] = 0.20
                position = 1
            # Enter short in strong downtrend with volume and trend strength
            elif trend_down and trend_filter and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend breakdown or weak trend
            if (not trend_up) or (adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend breakdown or weak trend
            if (not trend_down) or (adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DailyEMA50_ADX25_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0