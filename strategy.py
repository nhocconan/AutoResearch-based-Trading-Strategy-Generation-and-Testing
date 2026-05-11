#!/usr/bin/env python3
"""
4h_MoneyFlow_Index_Divergence_1dTrend
Hypothesis: Use Money Flow Index (MFI) divergence with 1d trend filter to identify reversals. Works in bull/bear by aligning with higher timeframe trend while using MFI divergence for entry timing. Target: 20-40 trades/year on 4h.
"""

name = "4h_MoneyFlow_Index_Divergence_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Money Flow Index (14) ===
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    positive_flow = np.where(typical_price > np.roll(typical_price, 1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < np.roll(typical_price, 1), raw_money_flow, 0)
    
    # Handle first element
    positive_flow[0] = 0
    negative_flow[0] = 0
    
    # Calculate money flow ratio
    pos_sum = pd.Series(positive_flow).rolling(window=14, min_periods=14).sum().values
    neg_sum = pd.Series(negative_flow).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    money_flow_ratio = np.where(neg_sum != 0, pos_sum / neg_sum, 100)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    # === Price and MFI Peaks/Troughs for Divergence ===
    # Find local peaks and troughs
    def find_peaks(arr, window=5):
        """Find local peaks"""
        peaks = np.zeros_like(arr, dtype=bool)
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                peaks[i] = True
        return peaks
    
    def find_troughs(arr, window=5):
        """Find local troughs"""
        troughs = np.zeros_like(arr, dtype=bool)
        for i in range(window, len(arr) - window):
            if arr[i] == np.min(arr[i-window:i+window+1]):
                troughs[i] = True
        return troughs
    
    price_peaks = find_peaks(close, window=3)
    price_troughs = find_troughs(close, window=3)
    mfi_peaks = find_peaks(mfi, window=3)
    mfi_troughs = find_troughs(mfi, window=3)
    
    # === Signal Parameters ===
    base_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers MFI calculation)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(mfi[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, MFI makes higher low
            if (price_troughs[i] and 
                i >= 10):  # Need to look back for confirmation
                # Find recent price trough
                lookback = min(i, 20)
                recent_price_troughs = np.where(price_troughs[i-lookback:i+1])[0]
                if len(recent_price_troughs) >= 2:
                    last_trough_idx = recent_price_troughs[-2]  # Second last trough
                    curr_trough_idx = recent_price_troughs[-1]  # Last trough
                    
                    # Check if current price trough is lower than previous
                    if close[curr_trough_idx] < close[last_trough_idx]:
                        # Check if MFI at current trough is higher than previous
                        if mfi[curr_trough_idx] > mfi[last_trough_idx]:
                            # Additional confirmation: MFI < 30 (oversold) and uptrend
                            if mfi[i] < 30 and close[i] > ema34_1d_aligned[i]:
                                signals[i] = base_size
                                position = 1
            
            # Bearish divergence: price makes higher high, MFI makes lower high
            elif (price_peaks[i] and 
                  i >= 10):  # Need to look back for confirmation
                # Find recent price peak
                lookback = min(i, 20)
                recent_price_peaks = np.where(price_peaks[i-lookback:i+1])[0]
                if len(recent_price_peaks) >= 2:
                    last_peak_idx = recent_price_peaks[-2]  # Second last peak
                    curr_peak_idx = recent_price_peaks[-1]  # Last peak
                    
                    # Check if current price peak is higher than previous
                    if close[curr_peak_idx] > close[last_peak_idx]:
                        # Check if MFI at current peak is lower than previous
                        if mfi[curr_peak_idx] < mfi[last_peak_idx]:
                            # Additional confirmation: MFI > 70 (overbought) and downtrend
                            if mfi[i] > 70 and close[i] < ema34_1d_aligned[i]:
                                signals[i] = -base_size
                                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: MFI crosses above 70 or trend change
                if mfi[i] > 70 or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size  # maintain position
            elif position == -1:
                # Exit short: MFI crosses below 30 or trend change
                if mfi[i] < 30 or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
    
    return signals