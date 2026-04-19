#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action near 1-day VWAP with volume confirmation and ATR-based stops.
# Uses 1-day VWAP as dynamic support/resistance, works in both trending and ranging markets.
# Targets 20-40 trades/year to avoid fee drag, with clear entry/exit rules.
name = "4h_1d_VWAP_Bounce_Volume_ATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP: typical price * volume / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vp = typical_price * df_1d['volume']
    cum_vp = vp.cumsum()
    cum_vol = df_1d['volume'].cumsum()
    vwap = (cum_vp / cum_vol).replace([np.inf, -np.inf], np.nan).ffill().values
    
    # Align 1d VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # 4h ATR for volatility and stop calculation (14-period)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(vwap_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vwap = vwap_aligned[i]
        atr_val = atr[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: bounce off VWAP support with volume
            if price > vwap and price < vwap + 0.5 * atr_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: rejection at VWAP resistance with volume
            elif price < vwap and price > vwap - 0.5 * atr_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price moves below VWAP or ATR stop
            if price < vwap or price < vwap + 2.0 * atr_val:  # trailing stop from entry area
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price moves above VWAP or ATR stop
            if price > vwap or price > vwap - 2.0 * atr_val:  # trailing stop from entry area
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals