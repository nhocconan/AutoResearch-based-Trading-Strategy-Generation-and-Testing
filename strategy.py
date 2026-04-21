#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly close for trend ===
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # === Daily range for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR(14) percentile (100-day lookback for regime)
    atr_percentile = pd.Series(atr_14).rolling(window=100, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align ATR percentile to 4h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # === Daily volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(sma_20_1w_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_trend = sma_20_1w_aligned[i]
        atr_percentile_val = atr_percentile_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (range) + weekly uptrend + volume
            if (atr_percentile_val < 30 and  # Low volatility regime
                price_close > weekly_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility (range) + weekly downtrend + volume
            elif (atr_percentile_val < 30 and   # Low volatility regime
                  price_close < weekly_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or opposite condition
            if position == 1 and (atr_percentile_val > 70 or price_close < weekly_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (atr_percentile_val > 70 or price_close > weekly_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WeeklyTrend_ATRVolatility_Volume"
timeframe = "4h"
leverage = 1.0