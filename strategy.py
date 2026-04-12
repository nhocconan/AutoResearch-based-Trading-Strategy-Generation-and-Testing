# 12h_1w_vwap_mean_reversion
# Strategy uses 1-week VWAP as mean price with Bollinger Bands for overextension detection.
# Mean reversion when price deviates >2σ from VWAP with volume confirmation.
# Works in both bull and bear markets as VWAP represents fair value and deviations correct over time.
# Low trade frequency expected due to strict deviation threshold and volume filter.

name = "12h_1w_vwap_mean_reversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for VWAP and Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # VWAP calculation (20-period)
    vwap_length = 20
    typical_price = (high_1w + low_1w + close_1w) / 3
    vwap_numerator = (typical_price * volume_1w)
    vwap_denominator = volume_1w
    
    vwap = pd.Series(vwap_numerator).rolling(window=vwap_length, min_periods=vwap_length).sum().values / \
           pd.Series(vwap_denominator).rolling(window=vwap_length, min_periods=vwap_length).sum().values
    
    # Standard deviation of price from VWAP
    price_diff = typical_price - vwap
    variance = pd.Series(price_diff * price_diff).rolling(window=vwap_length, min_periods=vwap_length).sum().values / \
               pd.Series(vwap_denominator).rolling(window=vwap_length, min_periods=vwap_length).sum().values
    std_dev = np.sqrt(np.maximum(variance, 0))
    
    # Upper and lower bands (2 standard deviations)
    upper_band = vwap + (2.0 * std_dev)
    lower_band = vwap - (2.0 * std_dev)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_1w).rolling(window=vwap_length, min_periods=vwap_length).mean().values
    vol_confirm = volume_1w > (vol_ma * 1.5)
    
    # Align VWAP, bands, and volume confirmation to 12h
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1w, vol_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price below lower band with volume confirmation
        if close[i] < lower_aligned[i] and vol_confirm_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price above upper band with volume confirmation
        elif close[i] > upper_aligned[i] and vol_confirm_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to VWAP
        elif position == 1 and close[i] >= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= vwap_aligned[i]:
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