#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 12h trend filter and volume confirmation.
# Uses 12h EMA for trend direction and 4h Camarilla levels for mean reversion entries.
# Volume confirmation ensures breakouts/reversals have conviction.
# Designed to work in both bull and bear markets by combining trend following (12h EMA)
# with mean reversion (Camarilla reversals) and volume filtering.
# Target: 80-150 total trades over 4 years (20-37/year) to balance opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h5 = []
    camarilla_h4 = []
    camarilla_h3 = []
    camarilla_l3 = []
    camarilla_l4 = []
    camarilla_l5 = []
    
    for i in range(len(high_1d)):
        if i == 0:
            camarilla_h5.append(np.nan)
            camarilla_h4.append(np.nan)
            camarilla_h3.append(np.nan)
            camarilla_l3.append(np.nan)
            camarilla_l4.append(np.nan)
            camarilla_l5.append(np.nan)
        else:
            # Previous day's OHLC
            phigh = high_1d[i-1]
            plow = low_1d[i-1]
            pclose = close_1d[i-1]
            range_val = phigh - plow
            
            camarilla_h5.append(pclose + range_val * 1.1 / 2)
            camarilla_h4.append(pclose + range_val * 1.1 / 4)
            camarilla_h3.append(pclose + range_val * 1.1 / 6)
            camarilla_l3.append(pclose - range_val * 1.1 / 6)
            camarilla_l4.append(pclose - range_val * 1.1 / 4)
            camarilla_l5.append(pclose - range_val * 1.1 / 2)
    
    camarilla_h5 = np.array(camarilla_h5)
    camarilla_h4 = np.array(camarilla_h4)
    camarilla_h3 = np.array(camarilla_h3)
    camarilla_l3 = np.array(camarilla_l3)
    camarilla_l4 = np.array(camarilla_l4)
    camarilla_l5 = np.array(camarilla_l5)
    
    # Align Camarilla levels to 4h timeframe
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(h5_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(l5_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition
        volume_condition = volume[i] > (volume_ma[i] * 1.5)
        
        # Trend condition: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Look for reversals at Camarilla levels with volume and trend alignment
            # Long when price touches L3/L4 in uptrend with volume
            # Short when price touches H3/H4 in downtrend with volume
            long_condition = (close[i] <= l3_aligned[i] * 1.002 or close[i] <= l4_aligned[i] * 1.002) and \
                           uptrend and volume_condition
            short_condition = (close[i] >= h3_aligned[i] * 0.998 or close[i] >= h4_aligned[i] * 0.998) and \
                            downtrend and volume_condition
            
            if long_condition:
                position = 1
                signals[i] = position_size
            elif short_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches H3 or H4 (profit target) or breaks below L5 (stop)
            if (close[i] >= h3_aligned[i] * 0.998 or close[i] >= h4_aligned[i] * 0.998 or
                close[i] <= l5_aligned[i] * 1.002):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches L3 or L4 (profit target) or breaks above H5 (stop)
            if (close[i] <= l3_aligned[i] * 1.002 or close[i] <= l4_aligned[i] * 1.002 or
                close[i] >= h5_aligned[i] * 0.998):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_1d_Camarilla_Reversal_Trend_Volume"
timeframe = "4h"
leverage = 1.0