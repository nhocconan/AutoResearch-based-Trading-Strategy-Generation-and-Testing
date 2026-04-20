#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for regime and signal generation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR(14) for volatility measurement and stop loss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily ATR(20) moving average for volatility regime filter
    atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    # Daily volume ratio (current / 20-period average) for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Daily price range for breakout detection
    daily_range = high_1d - low_1d
    range_ma_10 = pd.Series(daily_range).rolling(window=10, min_periods=10).mean().values
    range_ma_10_aligned = align_htf_to_ltf(prices, df_1d, range_ma_10)
    
    # 12h price data (primary timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(range_ma_10_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_14_1d_aligned[i]
        atr_ma_20 = atr_ma_20_1d_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        daily_range_val = daily_range[i] if i < len(daily_range) else 0
        range_ma_10_val = range_ma_10_aligned[i]
        
        # Volatility filter: avoid extremely low or high volatility
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_confirm = vol_ratio > 1.4
        
        # Breakout signal: today's range exceeds recent average range
        range_breakout = daily_range_val > 1.5 * range_ma_10_val
        
        # Direction based on close vs open of current 12h bar
        # We need to get the open price for the current bar
        open_price = prices['open'].values[i] if i < len(prices['open'].values) else price
        bullish = price > open_price
        bearish = price < open_price
        
        if position == 0:
            # Enter long on bullish breakout with volume and volatility confirmation
            if bullish and vol_confirm and vol_filter and range_breakout:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish breakdown with volume and volatility confirmation
            elif bearish and vol_confirm and vol_filter and range_breakout:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish reversal or volatility expansion
            if bearish or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish reversal or volatility expansion
            if bullish or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RangeBreakout_VolumeVolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0