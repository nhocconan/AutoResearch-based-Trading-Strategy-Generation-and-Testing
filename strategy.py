# 12h_1d_Wick_Reversal_Volume_Filter
# Wick reversal pattern with 1d trend filter and volume confirmation
# Long: bullish wick (long lower shadow) in uptrend + volume spike
# Short: bearish wick (long upper shadow) in downtrend + volume spike
# Exit: trend reversal or wick failure
# Designed for low trade frequency (<30/year) to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Wick_Reversal_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d: EMA50 trend filter ===
    close_1d = df_1d['close'].values
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Wick calculations
    body_size = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    total_range = high - low
    
    # Avoid division by zero
    lower_wick_ratio = np.where(total_range > 0, lower_wick / total_range, 0)
    upper_wick_ratio = np.where(total_range > 0, upper_wick / total_range, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        open_val = open_[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        lower_wick_ratio_val = lower_wick_ratio[i]
        upper_wick_ratio_val = upper_wick_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish wick (long lower shadow) in uptrend + volume spike
            if (lower_wick_ratio_val > 0.6 and                  # Strong lower wick
                upper_wick_ratio_val < 0.3 and                  # Small upper wick
                close_val > ema_val and                         # Price above EMA50 (uptrend)
                vol_ratio_val > 1.5):                           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: bearish wick (long upper shadow) in downtrend + volume spike
            elif (upper_wick_ratio_val > 0.6 and                # Strong upper wick
                  lower_wick_ratio_val < 0.3 and                # Small lower wick
                  close_val < ema_val and                       # Price below EMA50 (downtrend)
                  vol_ratio_val > 1.5):                         # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or wick failure
            if (close_val < ema_val or                    # Price below EMA50 (trend change)
                upper_wick_ratio_val > 0.5):              # Bearish wick forming
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or wick failure
            if (close_val > ema_val or                    # Price above EMA50 (trend change)
                lower_wick_ratio_val > 0.5):              # Bullish wick forming
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Fix: open_ should be defined
open_ = prices['open'].values if 'open' in prices.columns else np.zeros(n)