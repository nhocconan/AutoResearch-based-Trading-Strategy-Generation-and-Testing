#!/usr/bin/env python3
"""
4h_1d_Combined_RSI_CCI_Momentum_v1
Hypothesis: Combines daily RSI and CCI with 4h momentum to capture trend reversals in both bull and bear markets.
Uses RSI<30 and CCI<-100 for long entries, RSI>70 and CCI>100 for short entries, with volume confirmation.
Designed for lower trade frequency (target: 20-30 trades/year) by requiring multiple confluence factors.
Works in bull markets by catching pullbacks and in bear markets by selling rallies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Combined_RSI_CCI_Momentum_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for RSI and CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = rsi_4h.fillna(50).values  # Neutral when undefined
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.fillna(50).values
    
    # Calculate daily CCI(20)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_ma_20 = typical_price_1d.rolling(window=20, min_periods=20).mean()
    tp_std_20 = typical_price_1d.rolling(window=20, min_periods=20).std()
    cci_1d = (typical_price_1d - tp_ma_20) / (0.015 * tp_std_20)
    cci_1d = cci_1d.fillna(0).values
    
    # Align daily indicators to 4h timeframe (wait for daily close)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_4h[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(cci_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Momentum filter: 4h RSI not in extreme territory (avoid chasing)
        momentum_filter = (rsi_4h[i] > 30) and (rsi_4h[i] < 70)
        
        # Entry conditions: daily RSI and CCI extremes
        long_entry = (rsi_1d_aligned[i] < 30) and (cci_1d_aligned[i] < -100) and volume_filter and momentum_filter
        short_entry = (rsi_1d_aligned[i] > 70) and (cci_1d_aligned[i] > 100) and volume_filter and momentum_filter
        
        # Exit conditions: return to neutral levels
        long_exit = (rsi_1d_aligned[i] > 50) or (cci_1d_aligned[i] > 0)
        short_exit = (rsi_1d_aligned[i] < 50) or (cci_1d_aligned[i] < 0)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals