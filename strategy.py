#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1w EMA trend filter and volume confirmation
    # Long: Close breaks above H3 pivot AND 1w EMA50 rising AND volume > 1.5x avg
    # Short: Close breaks below L3 pivot AND 1w EMA50 falling AND volume > 1.5x avg
    # Exit: Close reverts to H4/L4 pivot or volume dry-up
    # Using 12h timeframe for low trade frequency, Camarilla for intraday structure,
    # 1w EMA50 for trend regime filter, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # H4, H3, H2, H1, L1, L2, L3, L4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4 = close + 1.5*(high-low)*1.1/2, H3 = close + 1.1*(high-low), etc.
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        high_val = high_1d[i-1]
        low_val = low_1d[i-1]
        close_val = close_1d[i-1]
        diff = high_val - low_val
        
        camarilla_h4[i] = close_val + 1.5 * diff * 1.1 / 2
        camarilla_h3[i] = close_val + 1.1 * diff
        camarilla_l3[i] = close_val - 1.1 * diff
        camarilla_l4[i] = close_val - 1.5 * diff * 1.1 / 2
    
    # Align daily Camarilla levels to 12h
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA rising/falling
        ema_rising = ema_1w_aligned[i] > ema_1w_aligned[i-1]
        ema_falling = ema_1w_aligned[i] < ema_1w_aligned[i-1]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + trend + volume
        long_entry = (close[i] > h3_12h[i]) and ema_rising and vol_confirm
        short_entry = (close[i] < l3_12h[i]) and ema_falling and vol_confirm
        
        # Exit logic: Close reverts to H4/L4 or volume dry-up
        long_exit = (close[i] < h4_12h[i]) or not vol_confirm
        short_exit = (close[i] > l4_12h[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0