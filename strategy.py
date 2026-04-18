#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for calculations
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h ATR (14-period) for volatility
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.roll(df_4h['close'], 1))
    tr3 = np.abs(df_4h['low'] - np.roll(df_4h['close'], 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA200 for long-term trend
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr1_1d = df_1d['high'] - df_1d['low']
    tr2_1d = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3_1d = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Bollinger Bands (20, 2.0)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_4h_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(sma_20[i]) or
            np.isnan(std_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 20-period average
        if i >= 20:
            atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
            atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
            vol_filter = not np.isnan(atr_ma_1d_aligned[i]) and atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        else:
            vol_filter = False
        
        trade_allowed = vol_filter
        
        if position == 0:
            # Long: Price touches lower BB with long-term uptrend
            if trade_allowed and close[i] <= lower_band[i] and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper BB with long-term downtrend
            elif trade_allowed and close[i] >= upper_band[i] and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches middle band or reverses to upper band
            if close[i] >= sma_20[i] or close[i] >= upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches middle band or reverses to lower band
            if close[i] <= sma_20[i] or close[i] <= lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ATR_VolumeFilter_BollingerMeanReversion_v1"
timeframe = "4h"
leverage = 1.0