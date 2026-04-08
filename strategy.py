#!/usr/bin/env python3
# 4h_1w_12h_adaptive_mean_reversion_v1
# Hypothesis: Mean reversion from extreme 12h RSI with weekly trend filter and volume confirmation.
# In weekly uptrend: buy when 12h RSI < 30 and price > weekly VWAP with volume surge.
# In weekly downtrend: sell when 12h RSI > 70 and price < weekly VWAP with volume surge.
# Exit when RSI returns to neutral (40-60) or weekly trend fails.
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_12h_adaptive_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly VWAP for trend filter
    df_1w = get_htf_data(prices, '1w')
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w = vwap_1w.replace(0, np.nan).ffill().bfill().values
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # 12h RSI for mean reversion signals
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate RSI(14) on 12h data
    delta = pd.Series(close_12h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.fillna(50).values
    
    # Align 12h RSI to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or weekly trend fails (price < VWAP)
            if rsi_12h_aligned[i] > 50 or close[i] < vwap_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or weekly trend fails (price > VWAP)
            if rsi_12h_aligned[i] < 50 or close[i] > vwap_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 30 with volume surge and weekly uptrend (price > VWAP)
            if (rsi_12h_aligned[i] < 30 and vol_surge and 
                close[i] > vwap_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 with volume surge and weekly downtrend (price < VWAP)
            elif (rsi_12h_aligned[i] > 70 and vol_surge and 
                  close[i] < vwap_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals