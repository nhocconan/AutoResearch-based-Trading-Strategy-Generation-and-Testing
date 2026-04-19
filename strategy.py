#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price position relative to 1d VWAP with volume confirmation
# In bull markets: price above VWAP indicates institutional accumulation -> long
# In bear markets: price below VWAP indicates institutional distribution -> short
# Volume spike confirms institutional participation
# Target: 15-25 trades/year to minimize fee drag in ranging 2025 market
name = "6h_VWAP_Position_Volume_Confirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP (typical price * volume / cumulative volume)
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    pv = typical_price * df_1d['volume'].values
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 6h ATR for volatility filtering
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(atr_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_6h[i]
        
        # Volume filter: current volume > 2.0x average volume (20-period) for institutional confirmation
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 2.0 * avg_volume
        
        # Price position relative to VWAP
        price_above_vwap = price > vwap_1d_aligned[i]
        price_below_vwap = price < vwap_1d_aligned[i]
        
        if position == 0:
            # Long: price above VWAP + volume spike
            if price_above_vwap and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP + volume spike
            elif price_below_vwap and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below VWAP or volatility expansion
            if not price_above_vwap or atr > 2.0 * np.nanmedian(atr_6h[max(0, i-20):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above VWAP or volatility expansion
            if not price_below_vwap or atr > 2.0 * np.nanmedian(atr_6h[max(0, i-20):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals