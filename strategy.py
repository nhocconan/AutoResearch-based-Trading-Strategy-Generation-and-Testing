#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla pivot with 1d volume confirmation and ATR volatility filter
# Uses Camarilla levels from daily timeframe for structure, volume confirmation for conviction,
# and volatility filter to avoid chop. Works in both bull (breakouts) and bear (mean reversion at levels)
# Target: 20-50 trades/year to minimize fee drag
name = "4h_camarilla_pivot_1d_volume_atr_v1"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    # Based on previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day: use same values (will be overwritten quickly)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Calculate Camarilla levels for each day
    # Formulas based on previous day's range
    range_1d = prev_high - prev_low
    # Avoid division by zero
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    
    # Camarilla levels
    # L4 = Close - 1.1 * Range / 2
    # L3 = Close - 1.1 * Range / 4
    # L2 = Close - 1.1 * Range / 6
    # L1 = Close - 1.1 * Range / 12
    # H1 = Close + 1.1 * Range / 12
    # H2 = Close + 1.1 * Range / 6
    # H3 = Close + 1.1 * Range / 4
    # H4 = Close + 1.1 * Range / 2
    
    L4 = prev_close - 1.1 * range_1d / 2
    L3 = prev_close - 1.1 * range_1d / 4
    L2 = prev_close - 1.1 * range_1d / 6
    L1 = prev_close - 1.1 * range_1d / 12
    H1 = prev_close + 1.1 * range_1d / 12
    H2 = prev_close + 1.1 * range_1d / 6
    H3 = prev_close + 1.1 * range_1d / 4
    H4 = prev_close + 1.1 * range_1d / 2
    
    # Align to 4h timeframe (each day = 6 bars of 4h)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    L2_4h = align_htf_to_ltf(prices, df_1d, L2)
    L1_4h = align_htf_to_ltf(prices, df_1d, L1)
    H1_4h = align_htf_to_ltf(prices, df_1d, H1)
    H2_4h = align_htf_to_ltf(prices, df_1d, H2)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    
    # Calculate daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after enough data for indicators
        # Skip if required data not available
        if (np.isnan(L1_4h[i]) or np.isnan(H1_4h[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        if position == 1:  # Long position
            # Exit: price touches H3 or H4 (strong resistance) OR volatility drops
            if close[i] >= H3_4h[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches L3 or L4 (strong support) OR volatility drops
            if close[i] <= L3_4h[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above H1 with volume confirmation and volatility filter
            if close[i] > H1_4h[i] and vol_confirm and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L1 with volume confirmation and volatility filter
            elif close[i] < L1_4h[i] and vol_confirm and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals