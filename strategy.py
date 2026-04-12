#!/usr/bin/env python3
"""
4h_1d_RSI_Contrarian_Confluence_v1
Hypothesis: Combine daily RSI extremes with 4h price action for mean-reversion entries. Long when daily RSI < 30 and price closes above 4h VWAP with volume confirmation; short when daily RSI > 70 and price closes below 4h VWAP. Uses daily RSI as a regime filter to avoid counter-trend trades in strong trends. Targets 20-40 trades/year by requiring confluence of daily extreme RSI, 4h VWAP close, and volume spike. Works in bull markets via oversold bounces and bear via overbought reversals, while avoiding chop via RSI neutrality (30-70).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Contrarian_Confluence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP using cumulative method (resets daily)
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = cum_pv / cum_vol
    
    # Daily RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Warmup for RSI and VWAP stability
        # Skip if any data invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Conditions
        oversold = rsi_1d_aligned[i] < 30
        overbought = rsi_1d_aligned[i] > 70
        vwap_break_up = close[i] > vwap[i]
        vwap_break_down = close[i] < vwap[i]
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        long_entry = oversold and vwap_break_up and volume_spike
        short_entry = overbought and vwap_break_down and volume_spike
        
        # Exit: RSI returns to neutral zone (40-60) or opposite extreme
        long_exit = rsi_1d_aligned[i] > 40  # Exit when RSI rises above 40
        short_exit = rsi_1d_aligned[i] < 60  # Exit when RSI falls below 60
        
        # Signal logic
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals