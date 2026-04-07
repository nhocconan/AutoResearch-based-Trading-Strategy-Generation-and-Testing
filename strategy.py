#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal with 1-day volume confirmation
# Long when price retraces to S3 (1.118) level with volume > 1.5x 24-period average
# Short when price retraces to R3 (1.118) level with volume > 1.5x 24-period average
# Exit when price reaches opposite H4/L4 level or closes beyond H3/L3
# Stoploss at 2 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day Camarilla levels for mean reversion in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_camarilla_reversal_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_h4 = prev_close + 1.5 * range_1d
    camarilla_l4 = prev_close - 1.5 * range_1d
    camarilla_h3 = prev_close + 1.25 * range_1d
    camarilla_l3 = prev_close - 1.25 * range_1d
    camarilla_h2 = prev_close + 1.166 * range_1d
    camarilla_l2 = prev_close - 1.166 * range_1d
    camarilla_h1 = prev_close + 1.083 * range_1d
    camarilla_l1 = prev_close - 1.083 * range_1d
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1-day volume average (24-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=24, min_periods=24).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches H4 or closes above H3
            elif close[i] >= h4_aligned[i] or close[i] > h3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches L4 or closes below L3
            elif close[i] <= l4_aligned[i] or close[i] < l3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Volume filter: volume > 1.5x 24-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: price at or below L3 with volume confirmation
            if close[i] <= l3_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price at or above H3 with volume confirmation
            elif close[i] >= h3_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals