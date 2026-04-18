#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian breakout and trend
    df_w = get_htf_data(prices, '1w')
    
    # Weekly Donchian breakout (20-period)
    dh = pd.Series(df_w['high']).rolling(window=20, min_periods=20).max().values
    dl = pd.Series(df_w['low']).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: price > weekly EMA50
    ema50_w = pd.Series(df_w['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly levels to daily
    dh_aligned = align_htf_to_ltf(prices, df_w, dh)
    dl_aligned = align_htf_to_ltf(prices, df_w, dl)
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Daily volume filter: current volume > 1.5 * 10-day average
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (1.5 * vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or
            np.isnan(ema50_w_aligned[i]) or np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        dh_val = dh_aligned[i]
        dl_val = dl_aligned[i]
        ema50_w_val = ema50_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above weekly Donchian high with trend and volume
            if close_val > dh_val and close_val > ema50_w_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with counter-trend and volume
            elif close_val < dl_val and close_val < ema50_w_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below weekly Donchian low
            if close_val < dl_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above weekly Donchian high
            if close_val > dh_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals