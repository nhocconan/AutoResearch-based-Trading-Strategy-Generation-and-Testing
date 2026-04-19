#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d trend filter and weekly volatility filter
# - 1d EMA(34) defines trend direction (long when price > EMA34, short when price < EMA34)
# - Weekly ATR(14) normalized by price for volatility regime filter (trade only when ATR/price > 0.02)
# - 12h price crossing above/below 1d EMA(34) with volume confirmation for entry
# - Exit on opposite cross or volatility collapse
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drift

name = "12h_EMA34_1dTrend_WeeklyVolFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get weekly data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w - low_1w, np.abs(high_1w - np.roll(close_1w, 1)))
    tr2 = np.maximum(np.abs(low_1w - np.roll(close_1w, 1)), tr1)
    tr2[0] = high_1w[0] - low_1w[0]  # first bar
    atr_14_1w = pd.Series(tr2).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    price_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    vol_norm = atr_1w_aligned / price_1w_aligned  # normalized volatility
    
    # 12h volume confirmation (20-period average)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_norm[i]) or 
            np.isnan(price_1w_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
            
        # Volatility filter: trade only when normalized ATR > 0.02 (2%)
        vol_filter = vol_norm[i] > 0.02
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = vol_ma_12h[i] > 0 and volume[i] > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Look for long entry: price crosses above 1d EMA34 + volume + volatility
            if close[i] > ema_34_1d_aligned[i] and close[i-1] <= ema_34_1d_aligned[i-1] and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price crosses below 1d EMA34 + volume + volatility
            elif close[i] < ema_34_1d_aligned[i] and close[i-1] >= ema_34_1d_aligned[i-1] and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on cross below EMA34 or volatility collapse
            if close[i] < ema_34_1d_aligned[i] or vol_norm[i] <= 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on cross above EMA34 or volatility collapse
            if close[i] > ema_34_1d_aligned[i] or vol_norm[i] <= 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals