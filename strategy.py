#!/usr/bin/env python3
"""
6h_ehlers_fisher_transform_1d_regime_v1
Hypothesis: Use Ehlers Fisher Transform on 6h for reversal signals, filtered by 1d trend via EMA200 and volatility regime via ATR ratio. 
Fisher turning points work well in ranging markets, while EMA200 filter avoids counter-trend trades in strong trends. 
ATR ratio (short/long) identifies low-volatility environments where Fisher signals are most reliable. 
Designed for low frequency (12-37 trades/year) to minimize fee impact while capturing mean-reversion opportunities.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ehlers_fisher_transform_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d ATR for volatility regime (using 14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Ehlers Fisher Transform on 6h prices
    # Price normalized to [-1, 1] range over lookback period
    lookback = 10
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1, price_range)
    # Normalized price: -1 to 1
    normalized_price = 2 * ((close - lowest_low) / price_range - 0.5)
    # Smooth normalized price
    smoothed_price = pd.Series(normalized_price).ewm(span=5, adjust=False).mean().values
    # Fisher Transform formula: 0.5 * ln((1+value)/(1-value))
    # Clip to avoid domain errors in log
    smoothed_price = np.clip(smoothed_price, -0.999, 0.999)
    fisher = 0.5 * np.log((1 + smoothed_price) / (1 - smoothed_price))
    # Further smooth Fisher
    fisher_smoothed = pd.Series(fisher).ewm(span=3, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after all warmup periods
        # Skip if regime data not available
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(fisher_smoothed[i]):
            signals[i] = 0.0
            continue
        
        # Regime filters:
        # 1. Only trade counter to daily trend when volatility is low (ATR ratio < 0.8)
        # 2. In high volatility (ATR ratio >= 0.8), follow the trend
        low_volatility = atr_ratio_1d_aligned[i] < 0.8
        above_ema200 = close[i] > ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit when Fisher crosses below -0.5 (mean reversion complete) or stop via opposing signal
            if fisher_smoothed[i] < -0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Fisher crosses above 0.5
            if fisher_smoothed[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry logic based on regime:
            # Low volatility: mean reversion (fade extremes)
            # High volatility: trend following
            if low_volatility:
                # Low volatility: fade Fisher extremes
                long_entry = (fisher_smoothed[i] < -1.0) and above_ema200  # Oversold but above long-term trend
                short_entry = (fisher_smoothed[i] > 1.0) and (not above_ema200)  # Overbought but below long-term trend
            else:
                # High volatility: follow Fisher crosses with trend
                long_entry = (fisher_smoothed[i] > -0.5 and fisher_smoothed[i-1] <= -0.5) and above_ema200
                short_entry = (fisher_smoothed[i] < 0.5 and fisher_smoothed[i-1] >= 0.5) and (not above_ema200)
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals