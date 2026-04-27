#!/usr/bin/env python3
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
    
    # Get weekly data for higher timeframe context (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA(21) for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate weekly volume moving average
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Precompute daily ATR(14) for position sizing
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr2_d[0] = tr1_d[0]
    tr3_d[0] = tr1_d[0]
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_14_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(atr_14_d[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA21
        price_above_ema = close[i] > ema_21_1w_aligned[i]
        price_below_ema = close[i] < ema_21_1w_aligned[i]
        
        # Volatility filter: avoid high volatility periods
        vol_filter = atr_14_d[i] < atr_14_1w_aligned[i]  # daily ATR less than weekly ATR
        
        # Volume filter: current volume above weekly average
        volume_filter = volume[i] > vol_ma_1w_aligned[i]
        
        # Long conditions: price above weekly EMA21 + low volatility + volume
        long_condition = (price_above_ema and 
                         vol_filter and 
                         volume_filter)
        
        # Short conditions: price below weekly EMA21 + low volatility + volume
        short_condition = (price_below_ema and 
                          vol_filter and 
                          volume_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or volatility spike
        elif position == 1 and (not price_above_ema or not vol_filter):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or not vol_filter):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA21_VolumeFilter_1wTrend"
timeframe = "1d"
leverage = 1.0