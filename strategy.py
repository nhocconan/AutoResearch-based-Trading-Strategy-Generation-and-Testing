#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend and structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA100 for trend filter
    close_1w = df_1w['close'].values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Weekly Donchian(30) channels for breakout signals
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high = pd.Series(high_1w).rolling(window=30, min_periods=30).max().values
    donch_low = pd.Series(low_1w).rolling(window=30, min_periods=30).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Volume confirmation: volume / 30-period average volume (weekly)
    vol_ma_30 = pd.Series(df_1w['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio_1w = df_1w['volume'].values / vol_ma_30
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_100_1w_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_100_1w_aligned[i]
        upper_band = donch_high_aligned[i]
        lower_band = donch_low_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high, uptrend, volume spike, moderate volatility
            if (price_close > upper_band and 
                price_close > ema_trend and 
                vol_ratio > 1.5 and 
                atr_ratio_val > 0.6 and atr_ratio_val < 2.0):
                signals[i] = 0.30
                position = 1
            # Enter short: price breaks below weekly Donchian low, downtrend, volume spike, moderate volatility
            elif (price_close < lower_band and 
                  price_close < ema_trend and 
                  vol_ratio > 1.5 and 
                  atr_ratio_val > 0.6 and atr_ratio_val < 2.0):
                signals[i] = -0.30
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility extremes
            if position == 1 and (price_close < lower_band or atr_ratio_val > 2.5 or atr_ratio_val < 0.4):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_band or atr_ratio_val > 2.5 or atr_ratio_val < 0.4):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1d_WeeklyDonchianBreakout_100Trend_VolumeATR"
timeframe = "1d"
leverage = 1.0