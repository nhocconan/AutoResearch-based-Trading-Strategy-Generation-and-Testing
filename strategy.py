#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivot levels (weekly)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous week
    prev_high = df_1w['high'].values
    prev_low = df_1w['low'].values
    prev_close = df_1w['close'].values
    
    # Camarilla formulas:
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.0 * (prev_high - prev_low)
    H2 = prev_close + 0.5 * (prev_high - prev_low)
    H1 = prev_close + 0.25 * (prev_high - prev_low)
    L1 = prev_close - 0.25 * (prev_high - prev_low)
    L2 = prev_close - 0.5 * (prev_high - prev_low)
    L3 = prev_close - 1.0 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align all levels to daily timeframe (shifted by 1 week for lookback)
    H4_1d = align_htf_to_ltf(prices, df_1w, H4)
    H3_1d = align_htf_to_ltf(prices, df_1w, H3)
    H2_1d = align_htf_to_ltf(prices, df_1w, H2)
    H1_1d = align_htf_to_ltf(prices, df_1w, H1)
    L1_1d = align_htf_to_ltf(prices, df_1w, L1)
    L2_1d = align_htf_to_ltf(prices, df_1w, L2)
    L3_1d = align_htf_to_ltf(prices, df_1w, L3)
    L4_1d = align_htf_to_ltf(prices, df_1w, L4)
    
    # Volume confirmation: volume > 1.5x 20-period average (daily)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any pivot level is not ready
        if (np.isnan(H4_1d[i]) or np.isnan(H3_1d[i]) or np.isnan(H2_1d[i]) or 
            np.isnan(H1_1d[i]) or np.isnan(L1_1d[i]) or np.isnan(L2_1d[i]) or 
            np.isnan(L3_1d[i]) or np.isnan(L4_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L2 (strong support broken) or volume spike fails
            if close[i] < L2_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H2 (strong resistance broken) or volume spike fails
            if close[i] > H2_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume must be present for any entry
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above H3 with volume (bullish breakout)
            # OR price bounces from L3/L4 with volume (bullish reversal)
            if ((close[i] > H3_1d[i] and close[i-1] <= H3_1d[i-1]) or  # Breakout above H3
                ((close[i] > L3_1d[i] and close[i-1] <= L3_1d[i-1]) or  # Bounce from L3
                 (close[i] > L4_1d[i] and close[i-1] <= L4_1d[i-1])) and  # Bounce from L4
                close[i] < H2_1d[i]):  # But not above strong resistance
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 with volume (bearish breakout)
            # OR price rejects from H3/H4 with volume (bearish reversal)
            elif ((close[i] < L3_1d[i] and close[i-1] >= L3_1d[i-1]) or  # Breakdown below L3
                  ((close[i] < H3_1d[i] and close[i-1] >= H3_1d[i-1]) or  # Rejection from H3
                   (close[i] < H4_1d[i] and close[i-1] >= H4_1d[i-1])) and  # Rejection from H4
                  close[i] > L2_1d[i]):  # But not below strong support
                position = -1
                signals[i] = -0.25
    
    return signals