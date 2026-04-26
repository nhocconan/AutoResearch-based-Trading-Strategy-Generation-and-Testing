#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v1
Hypothesis: Donchian(20) breakout with volume spike and choppiness regime filter on 4h timeframe.
In ranging markets (high chop), fade breakouts; in trending markets (low chop), follow breakouts.
Uses ATR-based trailing stop for risk control. Designed to work in both bull and bear markets
by adapting to regime conditions. Target: 20-50 trades per year per symbol.
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
    
    # Get 4h data for HTF trend filter (though we're already on 4h, we'll use it for alignment demonstration)
    # Actually, we'll compute everything on the 4h timeframe directly
    
    # Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR for volatility and stoploss
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_period = 20
    vol_ma = pd.Series(volume).rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index (CHOP) - measures whether market is choppy (ranging) or trending
    chop_period = 14
    # True Range
    tr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    # Highest high and lowest low over chop_period
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(chop_period)
    # CHOP > 61.8 = ranging/choppy market, CHOP < 38.2 = trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of Donchian (20), ATR (14), volume MA (20), CHOP (14)
    start_idx = max(donchian_period, atr_period, vol_ma_period, chop_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # Regime filters: only trade breakouts in trending markets (CHOP < 38.2)
        # In choppy markets (CHOP > 61.8), we could fade, but for simplicity we just avoid trading
        is_trending = chop_val < 38.2
        is_choppy = chop_val > 61.8
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume spike in trending market
            long_signal = (close_val > highest_high[i]) and vol_spike and is_trending
            
            # Short: Price breaks below Donchian lower band with volume spike in trending market
            short_signal = (close_val < lowest_low[i]) and vol_spike and is_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit: Price re-enters Donchian channel (below midpoint) OR trailing stop (2.5*ATR below high)
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if (close_val < donchian_mid) or (close_val < highest_since_entry - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit: Price re-enters Donchian channel (above midpoint) OR trailing stop (2.5*ATR above low)
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if (close_val > donchian_mid) or (close_val > lowest_since_entry + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0