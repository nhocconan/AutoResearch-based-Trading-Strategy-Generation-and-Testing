#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d data with volume confirmation and ATR filter.
# Camarilla levels provide precise support/resistance based on prior day's range.
# Works in both bull and bear markets by capturing reversions to mean at extreme levels.
# Volume confirmation ensures genuine interest at these levels.
# ATR filter avoids choppy markets where reversals fail.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Formula based on previous day's high, low, close
    camarilla_H4 = np.zeros(len(df_1d))
    camarilla_H3 = np.zeros(len(df_1d))
    camarilla_H2 = np.zeros(len(df_1d))
    camarilla_H1 = np.zeros(len(df_1d))
    camarilla_L1 = np.zeros(len(df_1d))
    camarilla_L2 = np.zeros(len(df_1d))
    camarilla_L3 = np.zeros(len(df_1d))
    camarilla_L4 = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        # Previous day's values
        phigh = df_1d['high'].iloc[i-1]
        plow = df_1d['low'].iloc[i-1]
        pclose = df_1d['close'].iloc[i-1]
        range_val = phigh - plow
        
        if range_val > 0:
            camarilla_H4[i] = pclose + range_val * 1.1 / 2
            camarilla_H3[i] = pclose + range_val * 1.1 / 4
            camarilla_H2[i] = pclose + range_val * 1.1 / 6
            camarilla_H1[i] = pclose + range_val * 1.1 / 12
            camarilla_L1[i] = pclose - range_val * 1.1 / 12
            camarilla_L2[i] = pclose - range_val * 1.1 / 6
            camarilla_L3[i] = pclose - range_val * 1.1 / 4
            camarilla_L4[i] = pclose - range_val * 1.1 / 2
        else:
            # Fallback if no range
            camarilla_H4[i] = camarilla_H3[i] = camarilla_H2[i] = camarilla_H1[i] = pclose
            camarilla_L1[i] = camarilla_L2[i] = camarilla_L3[i] = camarilla_L4[i] = pclose
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_H2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H2)
    camarilla_H1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    camarilla_L1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    camarilla_L2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L2)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Calculate ATR (14-period) for volatility filter
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(atr[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        atr_val = atr[i]
        
        # Get current Camarilla levels
        H4 = camarilla_H4_aligned[i]
        H3 = camarilla_H3_aligned[i]
        H2 = camarilla_H2_aligned[i]
        H1 = camarilla_H1_aligned[i]
        L1 = camarilla_L1_aligned[i]
        L2 = camarilla_L2_aligned[i]
        L3 = camarilla_L3_aligned[i]
        L4 = camarilla_L4_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # ATR filter: avoid extremely low volatility (chop)
        # Only trade when ATR is above its 50-period moving average
        if i >= 50:
            atr_ma = np.mean(atr[i-50:i])
            atr_filter = atr_val > 0.5 * atr_ma  # Avoid dead markets
        else:
            atr_filter = True  # Not enough data for MA, allow trading
        
        if position == 0:
            # Long: price touches or goes below L3/L4 with volume confirmation
            # This indicates potential bounce from strong support
            if ((price <= L3 or price <= L4) and 
                volume_confirm and 
                atr_filter):
                position = 1
                signals[i] = position_size
            # Short: price touches or goes above H3/H4 with volume confirmation
            # This indicates potential rejection from strong resistance
            elif ((price >= H3 or price >= H4) and 
                  volume_confirm and 
                  atr_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H1 (middle resistance) or volume drops significantly
            if (price >= H1 or vol < 0.4 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L1 (middle support) or volume drops significantly
            if (price <= L1 or vol < 0.4 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0