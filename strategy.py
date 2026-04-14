#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Reversal + Volume Spike + 1d Trend Filter
# Uses daily Camarilla pivot levels for mean reversion entries in 12h timeframe
# Volume confirmation ensures institutional participation
# 1d EMA50 filter ensures we trade against the daily trend only when momentum is exhausted
# Works in bull/bear by fading exhaustion moves at key pivot levels
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    high_low_range = prev_high - prev_low
    H4 = prev_close + 1.5 * high_low_range
    H3 = prev_close + 1.0 * high_low_range
    H2 = prev_close + 0.5 * high_low_range
    H1 = prev_close + 0.25 * high_low_range
    L1 = prev_close - 0.25 * high_low_range
    L2 = prev_close - 0.5 * high_low_range
    L3 = prev_close - 1.0 * high_low_range
    L4 = prev_close - 1.5 * high_low_range
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d EMA50 for trend filter (use previous day's EMA to avoid look-ahead)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().shift(1).values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x average volume (24-period = 12 days)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade against the daily trend
        # In uptrend (price > EMA50), look for short at resistance
        # In downtrend (price < EMA50), look for long at support
        if ema_50_aligned[i] > 0:  # Valid EMA value
            if price > ema_50_aligned[i]:
                # Daily uptrend - look for short at resistance levels
                if position == 0:
                    # Short at H3 or H4 with volume confirmation
                    if ((price >= H3_aligned[i] or price >= H4_aligned[i]) and 
                        vol > 1.5 * avg_vol[i]):
                        position = -1
                        signals[i] = -position_size
                    else:
                        signals[i] = 0.0
                elif position == 1:
                    # Exit long if price breaks above H4 (stop and reverse)
                    if price > H4_aligned[i]:
                        position = -1
                        signals[i] = -position_size
                    else:
                        signals[i] = position_size
                elif position == -1:
                    # Maintain short unless price breaks above H4 (stop)
                    if price > H4_aligned[i]:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -position_size
            else:
                # Daily downtrend - look for long at support levels
                if position == 0:
                    # Long at L3 or L4 with volume confirmation
                    if ((price <= L3_aligned[i] or price <= L4_aligned[i]) and 
                        vol > 1.5 * avg_vol[i]):
                        position = 1
                        signals[i] = position_size
                    else:
                        signals[i] = 0.0
                elif position == 1:
                    # Maintain long unless price breaks below L4 (stop)
                    if price < L4_aligned[i]:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = position_size
                elif position == -1:
                    # Exit short if price breaks below L4 (stop and reverse)
                    if price < L4_aligned[i]:
                        position = 1
                        signals[i] = position_size
                    else:
                        signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0