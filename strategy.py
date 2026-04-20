# 6h_1d_VWAP_Retrace_With_Volume
# Hypothesis: Price tends to revert to daily VWAP after strong moves, especially with volume confirmation.
# Works in both bull and bear markets as a mean-reversion edge on 6h timeframe.
# Uses 1d VWAP as dynamic support/resistance and volume spike for confirmation.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_VWAP_Retrace_With_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d VWAP Calculation (typical price * volume) / cumulative volume ===
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_num = (typical_price * df_1d['volume']).cumsum()
    vwap_den = df_1d['volume'].cumsum()
    vwap = vwap_num / vwap_den
    # Avoid division by zero on first bar
    vwap = vwap.replace(0, np.nan).ffill()
    vwap_values = vwap.values
    
    # Align VWAP to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # === 6h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 6h Price Change Rate (for momentum filter) ===
    close = prices['close'].values
    price_change = np.zeros_like(close)
    price_change[0] = 0
    for i in range(1, len(close)):
        price_change[i] = (close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        vwap_val = vwap_aligned[i]
        vol_ratio_val = vol_ratio[i]
        price_change_val = price_change[i]
        
        # Skip if any value is NaN
        if (np.isnan(vwap_val) or np.isnan(vol_ratio_val) or 
            np.isnan(price_change_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Avoid division by zero in VWAP calculation
        if vwap_val == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate deviation from VWAP as percentage
        dev_pct = (close_val - vwap_val) / vwap_val * 100
        
        if position == 0:
            # Long: Price below VWAP (oversold) with volume spike and negative momentum
            if dev_pct < -1.5 and vol_ratio_val > 2.0 and price_change_val < -0.005:
                signals[i] = 0.25
                position = 1
            # Short: Price above VWAP (overbought) with volume spike and positive momentum
            elif dev_pct > 1.5 and vol_ratio_val > 2.0 and price_change_val > 0.005:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to VWAP or volume dries up
            if dev_pct > -0.2 or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to VWAP or volume dries up
            if dev_pct < 0.2 or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals