#!/usr/bin/env python3
"""
4h_PriceChannel_Volume_Regime
Hypothesis: Trade breakouts from Donchian(20) and Keltner(20,1.5) channels on 4h timeframe with volume confirmation and Choppiness Index regime filter. Enter long when price breaks above upper channel with volume > 1.5x average and CHOP > 61.8 (ranging market). Enter short when price breaks below lower channel with volume confirmation and CHOP > 61.8. Uses ATR(20) for stop loss via signal=0 when price closes outside channel. Designed to capture breakouts in ranging markets while avoiding trends where breakouts fail. Works in bull/bear by focusing on mean-reversion breakouts in ranging conditions.
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
    
    # Donchian Channel (20)
    donchian_len = 20
    upper_dc = np.full(n, np.nan)
    lower_dc = np.full(n, np.nan)
    
    for i in range(donchian_len - 1, n):
        upper_dc[i] = np.max(high[i - donchian_len + 1:i + 1])
        lower_dc[i] = np.min(low[i - donchian_len + 1:i + 1])
    
    # Keltner Channel (20, 1.5)
    keltner_len = 20
    keltner_mult = 1.5
    atr = np.full(n, np.nan)
    keltner_upper = np.full(n, np.nan)
    keltner_lower = np.full(n, np.nan)
    
    if n >= keltner_len:
        # True Range
        tr = np.full(n, np.nan)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # ATR
        for i in range(keltner_len - 1, n):
            atr[i] = np.mean(tr[i - keltner_len + 1:i + 1])
        
        # Keltner
        for i in range(keltner_len - 1, n):
            ema_mid = np.mean(close[i - keltner_len + 1:i + 1])  # Simple MA for mid
            keltner_upper[i] = ema_mid + keltner_mult * atr[i]
            keltner_lower[i] = ema_mid - keltner_mult * atr[i]
    
    # Combined channel: use Donchian for breakout, Keltner for filtering
    upper_channel = upper_dc
    lower_channel = lower_dc
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_len = 20
    if n >= vol_len:
        for i in range(vol_len, n):
            vol_ma[i] = np.mean(volume[i - vol_len:i])
    
    # Choppiness Index (14) - regime filter
    chop_len = 14
    chop = np.full(n, np.nan)
    if n >= chop_len * 2:  # Need enough data for ATR sum
        atr_sum = np.full(n, np.nan)
        for i in range(chop_len - 1, n):
            atr_sum[i] = np.sum(tr[i - chop_len + 1:i + 1])
        
        max_hh = np.full(n, np.nan)
        min_ll = np.full(n, np.nan)
        for i in range(chop_len - 1, n):
            max_hh[i] = np.max(high[i - chop_len + 1:i + 1])
            min_ll[i] = np.min(low[i - chop_len + 1:i + 1])
        
        for i in range(chop_len - 1, n):
            if atr_sum[i] > 0 and (max_hh[i] - min_ll[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_hh[i] - min_ll[i])) / np.log10(chop_len)
    
    # Chop > 61.8 = ranging market (good for mean-reversion breakouts)
    chop_threshold = 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_len, keltner_len, vol_len, chop_len) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Chop filter: only trade in ranging markets
        chop_filter = chop[i] > chop_threshold
        
        if position == 0:
            # Long: price breaks above upper channel + volume + chop
            if close[i] > upper_channel[i] and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel + volume + chop
            elif close[i] < lower_channel[i] and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: hold until price closes below lower channel
            if close[i] < lower_channel[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: hold until price closes above upper channel
            if close[i] > upper_channel[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceChannel_Volume_Regime"
timeframe = "4h"
leverage = 1.0