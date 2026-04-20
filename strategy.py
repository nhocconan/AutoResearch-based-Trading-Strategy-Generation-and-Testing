#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h VWAP deviation with volume confirmation and ATR stop
# - Calculate 12h VWAP as dynamic mean reversion level
# - Long when price deviates below VWAP by >1.5 ATR with volume > 1.8x 20-period average
# - Short when price deviates above VWAP by >1.5 ATR with volume > 1.8x 20-period average
# - Exit when price returns to VWAP or ATR-based stop hit
# - Uses 12h VWAP (stable trend reference) with 4h execution
# - Target: 30-50 trades per year per symbol (120-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ATR for 12h (for deviation threshold)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate VWAP: cumulative(volume * price) / cumulative(volume)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    vp_12h = typical_price_12h * volume_12h
    cum_vp = np.nancumsum(vp_12h)
    cum_vol = np.nancumsum(volume_12h)
    vwap_12h = np.divide(cum_vp, cum_vol, out=np.full_like(cum_vp, np.nan), where=cum_vol!=0)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(vwap_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vwap = vwap_12h_aligned[i]
        atr = atr_12h_aligned[i]
        
        if position == 0:
            # Long entry: price below VWAP by >1.5 ATR + volume surge
            if price < (vwap - 1.5 * atr) and vol > 1.8 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price above VWAP by >1.5 ATR + volume surge
            elif price > (vwap + 1.5 * atr) and vol > 1.8 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to VWAP OR ATR stop hit (2*ATR below entry)
            if price >= vwap or price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP OR ATR stop hit (2*ATR above entry)
            if price <= vwap or price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_VWAP_Deviation_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0