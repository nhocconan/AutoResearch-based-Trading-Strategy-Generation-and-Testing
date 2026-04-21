#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === Daily ATR for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(20)
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) percentile (252-day lookback for regime)
    atr_percentile = pd.Series(atr_20).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align ATR percentile to 1h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # === Daily SMA50 for trend filter ===
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        atr_percentile_val = atr_percentile_aligned[i]
        sma_trend = sma_50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (range) + uptrend + volume
            if (atr_percentile_val < 30 and  # Low volatility regime
                price_close > sma_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.20
                position = 1
            # Enter short in low volatility (range) + downtrend + volume
            elif (atr_percentile_val < 30 and   # Low volatility regime
                  price_close < sma_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or opposite condition
            if position == 1 and (atr_percentile_val > 70 or price_close < sma_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (atr_percentile_val > 70 or price_close > sma_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_ATR_Volatility_Regime_SMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0