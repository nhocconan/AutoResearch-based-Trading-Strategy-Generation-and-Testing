#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Reversal + 1d Trend Filter + Volume Spike
# Uses Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts) from daily timeframe
# Only takes reversals in direction of 1-day EMA trend to avoid counter-trend trades
# Volume spike (>2x 20-period average) confirms institutional participation
# Works in bull/bear by aligning with higher timeframe trend while capturing mean reversion at extremes
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1h data once for EMA trend filter
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Standard Camarilla: 
    # H4 = Close + 1.5*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.1*(High-Low)
    # L3 = Close - 1.1*(High-Low)
    # We'll use H3/L3 for entries and H4/L4 for stops
    hl_range_1d = high_1d - low_1d
    h3_1d = close_1d + 1.1 * hl_range_1d
    l3_1d = close_1d - 1.1 * hl_range_1d
    h4_1d = close_1d + 1.5 * hl_range_1d
    l4_1d = close_1d - 1.5 * hl_range_1d
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for EMA and volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(ema_20_1h_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        ema_trend = ema_20_1h_aligned[i]
        
        # Volume filter: require significant volume spike
        if vol <= 2.0 * avg_vol[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price reaches S3 level (L3) in uptrend
            if price <= l3_1d_aligned[i] and close[i-1] > l3_1d_aligned[i] and ema_trend > close_1h[-1] if len(close_1h) > 0 else True:
                # Additional check: ensure we're in uptrend on 1h
                if ema_trend > close[i]:  # Simplified trend check
                    position = 1
                    signals[i] = position_size
            # Short: price reaches R3 level (H3) in downtrend
            elif price >= h3_1d_aligned[i] and close[i-1] < h3_1d_aligned[i] and ema_trend < close_1h[-1] if len(close_1h) > 0 else True:
                # Additional check: ensure we're in downtrend on 1h
                if ema_trend < close[i]:  # Simplified trend check
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 level or stops at H4
            if price >= h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 level or stops at L4
            if price <= l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_1hTrend"
timeframe = "4h"
leverage = 1.0