#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# Long when price breaks above Donchian(20) high AND volume > 1.8x 20-bar avg AND chop < 61.8 (trending)
# Short when price breaks below Donchian(20) low AND volume > 1.8x 20-bar avg AND chop < 61.8 (trending)
# Exit when price crosses opposite Donchian level (exit long on low, exit short on high)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing trends.
# Donchian channels provide clear structure for breakouts in both bull and bear markets.
# Volume confirmation ensures institutional participation, chop filter avoids whipsaws in ranging markets.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h.

name = "4h_Donchian20_VolumeConfirm_ChopFilter_v1"
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
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    # Choppiness Index (CHOP) - regime filter
    # CHOP(14) > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # We use CHOP < 61.8 to allow trending markets (avoid strong ranging)
    tr_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum()
    abs_close_change = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(tr_range / abs_close_change) / np.log10(14)
    chop = chop.values
    chop_filter = chop < 61.8  # Allow trending markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        curr_close = close[i]
        
        # Donchian levels
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low (trend reversal)
            if curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (trend reversal)
            if curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND volume confirmation AND chop filter
            if curr_close > upper and vol_conf and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND volume confirmation AND chop filter
            elif curr_close < lower and vol_conf and chop_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals