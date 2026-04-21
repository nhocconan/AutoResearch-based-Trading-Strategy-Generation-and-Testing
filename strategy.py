#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Bollinger Bands for volatility regime ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period SMA
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    # 20-period standard deviation
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    # Bollinger Bands
    upper_bb = sma_20_1d + (2 * std_20_1d)
    lower_bb = sma_20_1d - (2 * std_20_1d)
    
    # Bollinger Band Width as volatility measure
    bb_width = (upper_bb - lower_bb) / sma_20_1d
    
    # BB Width percentile (252-day lookback for regime)
    bb_width_percentile = pd.Series(bb_width).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align BB Width percentile to 12h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # === Daily SMA50 for trend filter ===
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bb_width_percentile_val = bb_width_percentile_aligned[i]
        sma_trend = sma_50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (range) + uptrend + volume
            if (bb_width_percentile_val < 30 and  # Low volatility regime (BB width low)
                price_close > sma_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility (range) + downtrend + volume
            elif (bb_width_percentile_val < 30 and   # Low volatility regime
                  price_close < sma_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or opposite condition
            if position == 1 and (bb_width_percentile_val > 70 or price_close < sma_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (bb_width_percentile_val > 70 or price_close > sma_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_BB_Width_Percentile_SMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0