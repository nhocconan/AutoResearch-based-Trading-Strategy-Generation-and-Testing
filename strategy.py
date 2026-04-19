#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RVI_BullishEngulfing_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RVI (Relative Vigor Index) on 4h
    # RVI = (Close - Open) / (High - Low) smoothed
    open_ = prices['open'].values
    numerator = close - open_
    denominator = high - low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    raw_rvi = numerator / denominator
    
    # Smooth RVI with 10-period SMA
    rvi = pd.Series(raw_rvi).rolling(window=10, min_periods=10).mean().values
    rvi_signal = pd.Series(rvi).rolling(window=4, min_periods=4).mean().values  # Signal line
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rvi[i]) or np.isnan(rvi_signal[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        # Trend filter: price relative to daily EMA34
        price_above_ema = price > ema34_1d_aligned[i]
        price_below_ema = price < ema34_1d_aligned[i]
        
        # Bullish engulfing pattern: current bullish candle engulfs previous bearish candle
        bullish_engulf = (close[i] > open_[i]) and (open_[i-1] > close[i-1]) and \
                         (close[i] >= open_[i-1]) and (open_[i] <= close[i-1])
        
        # Bearish engulfing pattern: current bearish candle engulfs previous bullish candle
        bearish_engulf = (close[i] < open_[i]) and (open_[i-1] < close[i-1]) and \
                         (close[i] <= open_[i-1]) and (open_[i] >= close[i-1])
        
        if position == 0:
            # Long: RVI crosses above signal line + bullish engulfing + volume + uptrend
            if rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1] and bullish_engulf and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: RVI crosses below signal line + bearish engulfing + volume + downtrend
            elif rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1] and bearish_engulf and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RVI crosses below signal line (momentum loss)
            if rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RVI crosses above signal line (momentum loss)
            if rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals