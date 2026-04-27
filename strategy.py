#!/usr/bin/env python3
"""
6h_OrderBookImbalance_Reversal_v1
Hypothesis: In 6B timeframe, extreme order book imbalances (proxy via volume-price divergence) 
signal short-term reversals. Buy when selling pressure exhausts (price down but volume up < average), 
sell when buying pressure exhausts (price up but volume down < average). 
Uses 1d trend filter to avoid counter-trend trades and 1w volatility filter to adapt position size.
Designed for low trade frequency (~15-25/year) to minimize fee drag in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume-price divergence: volume change vs price change
    # Normalized volume change: (V[t] - V[t-1]) / V[t-1]
    vol_change = np.diff(volume, prepend=volume[0])
    vol_change_pct = np.where(volume != 0, vol_change / volume, 0)
    
    # Price change
    price_change = np.diff(close, prepend=close[0])
    price_change_pct = np.where(close != 0, price_change / close, 0)
    
    # Divergence signal: when price and volume move opposite directions
    # Negative divergence (price down, volume up) = buying pressure
    # Positive divergence (price up, volume down) = selling pressure
    divergence = -price_change_pct * vol_change_pct  # Negative when price and volume move opposite
    
    # Smooth divergence to reduce noise
    div_smooth = pd.Series(divergence).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Extreme divergence thresholds (top/bottom 10%)
    div_pos_threshold = np.nanpercentile(div_smooth, 90)
    div_neg_threshold = np.nanpercentile(div_smooth, 10)
    
    # 1d trend filter: only trade in direction of higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1w volatility filter: reduce size in high volatility
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    atr_1w = pd.Series(high_1w - low_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1w = pd.Series(atr_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    # Volatility regime: high volatility when current ATR > 1.5 * MA ATR
    vol_regime = atr_1w > (1.5 * atr_ma_1w)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime)
    
    # Base position size
    base_size = 0.25
    
    signals = np.zeros(n)
    
    # Warmup
    start_idx = max(5, 50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(div_smooth[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr_ma_1w_aligned[i]) or 
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility-adjusted size
        size = base_size * 0.5 if vol_regime_aligned[i] else base_size
        
        div_val = div_smooth[i]
        close_val = close[i]
        ema50 = ema50_1d_aligned[i]
        
        # Determine trend
        uptrend = close_val > ema50
        downtrend = close_val < ema50
        
        if div_val > div_pos_threshold and downtrend:
            # Extreme selling pressure exhaustion -> potential long
            signals[i] = size
        elif div_val < div_neg_threshold and uptrend:
            # Extreme buying pressure exhaustion -> potential short
            signals[i] = -size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_OrderBookImbalance_Reversal_v1"
timeframe = "6h"
leverage = 1.0