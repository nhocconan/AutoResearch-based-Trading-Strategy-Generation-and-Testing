#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for price channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1d Donchian(20) channels
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w EMA(50) for higher timeframe trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Volume ratio (current / 20-period average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    
    # Align HTF indicators to lower timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Daily price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        ema_trend_1w = ema_50_1w_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # Trend filter: price above/below EMA(50) and aligned with weekly trend
        trend_up = price > ema_trend and ema_trend > ema_trend_1w
        trend_down = price < ema_trend and ema_trend < ema_trend_1w
        
        # Volatility filter: avoid extremely low or high volatility
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 4.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio > 1.5)
        
        if position == 0:
            # Long when price breaks above Donchian high + uptrend + volume
            if price > donch_high_val and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low + downtrend + volume
            elif price < donch_low_val and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend reversal
            if price < donch_low_val or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend reversal
            if price > donch_high_val or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_Trend_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0