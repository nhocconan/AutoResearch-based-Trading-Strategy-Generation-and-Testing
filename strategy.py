#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for intermediate trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily EMA(200) for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily volume ratio (current / 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 1h price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h ATR(14) for volatility filter
    tr1_1h = high - low
    tr2_1h = np.abs(high - np.roll(close, 1))
    tr3_1h = np.abs(low - np.roll(close, 1))
    tr2_1h[0] = tr1_1h[0]
    tr3_1h[0] = tr1_1h[0]
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    atr_14_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    
    # 1h volume ratio (current / 20-period average)
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1h = volume / np.where(vol_ma_20_1h == 0, 1, vol_ma_20_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(atr_14_1h[i]) or np.isnan(vol_ratio_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend_50 = ema_50_1d_aligned[i]
        ema_trend_200 = ema_200_1d_aligned[i]
        atr_daily = atr_14_1d_aligned[i]
        vol_ratio_daily = vol_ratio_1d_aligned[i]
        atr_1h = atr_14_1h[i]
        vol_ratio_1h = vol_ratio_1h[i]
        
        # Trend filter: price above/below both EMAs
        trend_up = price > ema_trend_50 and ema_trend_50 > ema_trend_200
        trend_down = price < ema_trend_50 and ema_trend_50 < ema_trend_200
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_1h > 0.5 * atr_daily
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_1h > 1.3) and (vol_ratio_daily > 1.2)
        
        if position == 0:
            # Enter long in strong uptrend with volume
            if trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Enter short in strong downtrend with volume
            elif trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend breakdown or volatility spike
            if (not trend_up) or (atr_1h > 3.0 * atr_daily):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend breakdown or volatility spike
            if (not trend_down) or (atr_1h > 3.0 * atr_daily):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_1d_EMA50_200_Trend_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0