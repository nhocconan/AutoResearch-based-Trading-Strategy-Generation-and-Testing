#!/usr/bin/env python3
"""
4h_Triple_RSI_Confluence_v1
4h strategy using multi-timeframe RSI confluence + volume confirmation.
- Long: 4h RSI(14) < 30 AND 12h RSI(14) < 40 AND 1d RSI(14) < 50 AND volume > 1.5x 20-period average
- Short: 4h RSI(14) > 70 AND 12h RSI(14) > 60 AND 1d RSI(14) > 50 AND volume > 1.5x 20-period average
- Exit: Opposite RSI condition or volume divergence
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in both bull and bear markets by buying oversold and selling overbought extremes
"""

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
    
    # Calculate 4h RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Get 12h data for RSI
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    delta_12h = np.diff(close_12h, prepend=close_12h[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    avg_gain_12h = pd.Series(gain_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_12h = pd.Series(loss_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_12h = avg_gain_12h / (avg_loss_12h + 1e-10)
    rsi_12h_raw = 100 - (100 / (1 + rs_12h))
    rsi_12h = align_htf_to_ltf(prices, df_12h, rsi_12h_raw)
    
    # Get 1d data for RSI and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d_raw = 100 - (100 / (1 + rs_1d))
    rsi_1d = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    # Volume confirmation (20-period average on 1d)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for RSI calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_4h[i]) or np.isnan(rsi_12h[i]) or np.isnan(rsi_1d[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # RSI conditions
        rsi_oversold = (rsi_4h[i] < 30) and (rsi_12h[i] < 40) and (rsi_1d[i] < 50)
        rsi_overbought = (rsi_4h[i] > 70) and (rsi_12h[i] > 60) and (rsi_1d[i] > 50)
        
        if position == 0:
            # Long: multi-timeframe RSI oversold + volume confirmation
            if rsi_oversold and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: multi-timeframe RSI overbought + volume confirmation
            elif rsi_overbought and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought on any timeframe or volume divergence
            if (rsi_4h[i] > 50) or (rsi_12h[i] > 50) or (rsi_1d[i] > 50) or (not vol_confirm):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold on any timeframe or volume divergence
            if (rsi_4h[i] < 50) or (rsi_12h[i] < 50) or (rsi_1d[i] < 50) or (not vol_confirm):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Triple_RSI_Confluence_v1"
timeframe = "4h"
leverage = 1.0