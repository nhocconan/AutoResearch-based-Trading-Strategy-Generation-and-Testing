#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_Pivot_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data ONCE for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 10 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # H4 = Close + 1.1/2 * (High - Low)
    # L4 = Close - 1.1/2 * (High - Low)
    # H3 = Close + 1.1/4 * (High - Low)
    # L3 = Close - 1.1/4 * (High - Low)
    # H2 = Close + 1.1/6 * (High - Low)
    # L2 = Close - 1.1/6 * (High - Low)
    # H1 = Close + 1.1/12 * (High - Low)
    # L1 = Close - 1.1/12 * (High - Low)
    camarilla_calc = (1.1 / 12) * (high_1d - low_1d)
    h1 = close_1d + camarilla_calc
    l1 = close_1d - camarilla_calc
    h2 = close_1d + camarilla_calc * 2
    l2 = close_1d - camarilla_calc * 2
    h3 = close_1d + camarilla_calc * 3
    l3 = close_1d - camarilla_calc * 3
    h4 = close_1d + camarilla_calc * 4
    l4 = close_1d - camarilla_calc * 4
    
    # Align all levels to 4h timeframe
    h1_4h = align_htf_to_ltf(prices, df_1d, h1)
    l1_4h = align_htf_to_ltf(prices, df_1d, l1)
    h2_4h = align_htf_to_ltf(prices, df_1d, h2)
    l2_4h = align_htf_to_ltf(prices, df_1d, l2)
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1w EMA34 for higher timeframe trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 4h volume average (20-period) for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    vol_avg_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    # 4h ATR for stop loss
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        h1_val = h1_4h[i]
        l1_val = l1_4h[i]
        h2_val = h2_4h[i]
        l2_val = l2_4h[i]
        h3_val = h3_4h[i]
        l3_val = l3_4h[i]
        h4_val = h4_4h[i]
        l4_val = l4_4h[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_avg = vol_avg_4h_aligned[i]
        current_atr = atr_4h_aligned[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(h1_val) or np.isnan(l1_val) or np.isnan(ema_trend) or 
            np.isnan(vol_avg) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5x 4h average volume
        vol_spike = current_volume > 1.5 * vol_avg
        
        if position == 0:
            # Long: price touches or breaks below L3 with volume spike in uptrend
            if current_close <= l3_val and current_close > l4_val and vol_spike and current_close > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price touches or breaks above H3 with volume spike in downtrend
            elif current_close >= h3_val and current_close < h4_val and vol_spike and current_close < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price crosses above H3 or stops below L4
            if current_close >= h3_val:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below L3 or stops above H4
            if current_close <= l3_val:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals