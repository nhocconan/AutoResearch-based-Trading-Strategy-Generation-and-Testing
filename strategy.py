#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# Long when price breaks above upper Donchian channel with volume spike
# Short when price breaks below lower Donchian channel with volume spike
# Uses proven Donchian breakout structure with volume confirmation to filter false breakouts
# ATR-based stoploss limits downside in bear markets (2022 crash protection)
# Target: 75-200 total trades over 4 years (19-50/year) for optimal fee drag balance

name = "4h_Donchian20_VolumeSpike_ATRStop_v1"
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
    
    # Donchian channel (20-period) - calculated on close prices
    lookback = 20
    upper_channel = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = max(lookback, 20, 14)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation to avoid false breakouts
            if curr_volume_confirm:
                # Bullish entry: price breaks above upper channel with volume
                if curr_high > curr_upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_stop = entry_price - 2.5 * curr_atr  # 2.5 ATR stoploss
                # Bearish entry: price breaks below lower channel with volume
                elif curr_low < curr_lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_stop = entry_price + 2.5 * curr_atr  # 2.5 ATR stoploss
        
        elif position == 1:  # Long position
            # Check stoploss or channel re-entry for exit
            if curr_low < atr_stop or curr_high < curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stoploss upward as price moves in our favor
                atr_stop = max(atr_stop, curr_close - 2.5 * curr_atr)
        
        elif position == -1:  # Short position
            # Check stoploss or channel re-entry for exit
            if curr_high > atr_stop or curr_low > curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stoploss downward as price moves in our favor
                atr_stop = min(atr_stop, curr_close + 2.5 * curr_atr)
    
    return signals