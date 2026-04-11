#!/usr/bin/env python3
"""
1h_4d_RSI_Range_Mean_Reversion
Hypothesis: In 1h timeframe, use 4-hour RSI to identify overbought/oversold conditions and mean-revert toward the 1-day VWAP. 
Works in both bull and bear markets by fading extreme RSI readings during ranging periods, avoiding strong trends.
Targets 20-30 trades/year per symbol with strict RSI thresholds and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_RSI_Range_Mean_Reversion"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Load 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h RSI(14) for entry confirmation
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs_1h))
    
    # Calculate 1-day VWAP (typical price * volume)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1h[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_filter = volume[i] > 1.2 * vol_ma_20[i]
        
        # Mean reversion conditions based on 4h RSI extremes
        rsi_overbought = rsi_4h_aligned[i] > 70
        rsi_oversold = rsi_4h_aligned[i] < 30
        
        # Price position relative to 1d VWAP
        price_above_vwap = close[i] > vwap_1d_aligned[i]
        price_below_vwap = close[i] < vwap_1d_aligned[i]
        
        # Entry conditions: fade RSI extremes toward VWAP
        long_entry = rsi_oversold and price_below_vwap and volume_filter
        short_entry = rsi_overbought and price_above_vwap and volume_filter
        
        # Exit conditions: RSI returns to neutral range or price crosses VWAP
        long_exit = (rsi_4h_aligned[i] > 50) or (close[i] > vwap_1d_aligned[i])
        short_exit = (rsi_4h_aligned[i] < 50) or (close[i] < vwap_1d_aligned[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals