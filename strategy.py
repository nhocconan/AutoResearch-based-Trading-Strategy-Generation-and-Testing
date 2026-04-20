#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly trend filter + Daily mean reversion at 1D support/resistance
# Uses weekly EMA to filter direction and daily RSI for mean reversion entries.
# Works in bull/bear by aligning with higher timeframe trend while capturing reversals.
# Target: 15-25 trades/year with tight entry conditions to minimize fee drag.

name = "1d_1w_EMA_RSI_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly EMA Trend Filter ===
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily RSI for Mean Reversion ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # RSI(14) calculation
    roll_up = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    roll_down = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        rsi_val = rsi_values[i]
        ema40_1w_val = ema40_1w_aligned[i]
        vol_ma20_val = vol_ma20[i]
        vol_val = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema40_1w_val) or 
            np.isnan(vol_ma20_val) or vol_ma20_val == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = vol_val / vol_ma20_val
        
        if position == 0:
            # Long: Uptrend (price > weekly EMA) + oversold RSI + volume confirmation
            if (close_val > ema40_1w_val and 
                rsi_val < 30 and 
                vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < weekly EMA) + overbought RSI + volume confirmation
            elif (close_val < ema40_1w_val and 
                  rsi_val > 70 and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or trend breaks
            if rsi_val > 50 or close_val < ema40_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or trend breaks
            if rsi_val < 50 or close_val > ema40_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals