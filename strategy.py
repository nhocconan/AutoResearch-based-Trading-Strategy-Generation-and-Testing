#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_WickRejection_Engulfing_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h range for wick analysis
    range_4h = high - low
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_) & (open_ < np.roll(close, 1)) & (close > np.roll(open_, 1)) & (np.roll(close, 1) < np.roll(open_, 1))
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close < open_) & (open_ > np.roll(close, 1)) & (close < np.roll(open_, 1)) & (np.roll(close, 1) > np.roll(open_, 1))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upw = upper_wick[i]
        loww = lower_wick[i]
        rng = range_4h[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Wick rejection: long wick relative to body
        body = abs(close[i] - open_[i])
        if body > 0:
            upper_wick_ratio = upw / body
            lower_wick_ratio = loww / body
        else:
            upper_wick_ratio = 0
            lower_wick_ratio = 0
        
        # Wick rejection conditions
        upper_wick_rej = upper_wick_ratio > 2.0  # Long upper wick
        lower_wick_rej = lower_wick_ratio > 2.0  # Long lower wick
        
        if position == 0:
            # Long: bullish engulfing + lower wick rejection + price above daily EMA200
            if bullish_engulf[i] and lower_wick_rej and price > ema200_1d_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing + upper wick rejection + price below daily EMA200
            elif bearish_engulf[i] and upper_wick_rej and price < ema200_1d_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish engulfing or upper wick rejection
            if bearish_engulf[i] or upper_wick_rej:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish engulfing or lower wick rejection
            if bullish_engulf[i] or lower_wick_rej:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals