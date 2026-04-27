#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeConfirm_ChopFilter_v1
Hypothesis: Donchian(20) breakouts with volume confirmation and choppiness regime filter capture strong trends while avoiding whipsaws in ranging markets. 
Volume confirmation ensures breakout validity, chop filter (CHOP > 61.8 = range) avoids false breakouts. 
ATR-based stoploss (2.5x) manages risk. Discrete sizing (0.25) controls fee drawdown. 
Works in bull/bear: breakouts capture momentum, chop filter avoids range whipsaws. 
Target: 75-200 trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian(20) channels: upper = 20-period high, lower = 20-period low
    # Use min_periods=20 to avoid look-ahead
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = ranging market (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first bar
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only allow breakouts when NOT strongly ranging
    
    # Align HTF indicators (volume_confirm and chop_filter are LTF but derived from HTF-aligned logic)
    # Actually, volume and chop are calculated on LTF, so no alignment needed.
    # But for consistency with MTF patterns, we'll align if they were HTF-derived.
    # Here, volume_confirm and chop_filter are LTF, so use directly.
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level to minimize churn)
    
    # Warmup: need Donchian (20), volume avg (20), chop (14)
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Long breakout: price > upper channel with volume and not choppy
            if close_val > upper and vol_conf and chop_ok:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short breakout: price < lower channel with volume and not choppy
            elif close_val < lower and vol_conf and chop_ok:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: stoploss (2.5*ATR) or price re-enters channel (middle)
            atr_approx = atr_14[i]
            stop_loss = entry_price - 2.5 * atr_approx
            middle = (upper + lower) / 2  # Donchian middle
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < middle:  # Re-entry to middle = trend weakness
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: stoploss (2.5*ATR) or price re-enters channel (middle)
            atr_approx = atr_14[i]
            stop_loss = entry_price + 2.5 * atr_approx
            middle = (upper + lower) / 2  # Donchian middle
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > middle:  # Re-entry to middle = trend weakness
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0