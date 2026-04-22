# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data (HTF) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR (14-period) for stop-loss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume average (20-period) - use same timeframe as price
    vol_avg_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day Donchian high with volume confirmation
            if (prices['close'].iloc[i] > donch_high_20_aligned[i] and 
                prices['volume'].iloc[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = prices['close'].iloc[i]
            # Short: Price breaks below 20-day Donchian low with volume confirmation
            elif (prices['close'].iloc[i] < donch_low_20_aligned[i] and 
                  prices['volume'].iloc[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = prices['close'].iloc[i]
        else:
            # Track highest/lowest price since entry for trailing stop
            if position == 1:
                # Long position: trail from highest high
                highest_since_entry = np.maximum(
                    donch_high_20_aligned[i],  # placeholder, will be updated below
                    np.max(prices['high'].iloc[:i+1]) if i > 0 else prices['high'].iloc[i]
                )
                # Actually track highest high since entry
                if i == 1:
                    highest_since_entry = prices['high'].iloc[i]
                else:
                    highest_since_entry = np.maximum(
                        highest_since_entry if 'highest_since_entry' in locals() else entry_price,
                        prices['high'].iloc[i]
                    )
                # Exit conditions: price reversal or trailing stop
                if (prices['close'].iloc[i] < donch_low_20_aligned[i] or  # reversal
                    prices['close'].iloc[i] < highest_since_entry - 2.0 * atr_14_aligned[i]):  # trailing stop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short position: trail from lowest low
                if i == 1:
                    lowest_since_entry = prices['low'].iloc[i]
                else:
                    lowest_since_entry = np.minimum(
                        lowest_since_entry if 'lowest_since_entry' in locals() else entry_price,
                        prices['low'].iloc[i]
                    )
                # Exit conditions: price reversal or trailing stop
                if (prices['close'].iloc[i] > donch_high_20_aligned[i] or  # reversal
                    prices['close'].iloc[i] > lowest_since_entry + 2.0 * atr_14_aligned[i]):  # trailing stop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian20_Volume_ATR_Trail"
timeframe = "12h"
leverage = 1.0