#!/usr/bin/env python3
"""
Experiment #3771: 6h Donchian(20) breakout + 1d volume profile high-volume node (VHN) + ATR filter
HYPOTHESIS: 6h Donchian breakouts capture intermediate swings, with 1d volume profile identifying high-volume nodes (HVN) as support/resistance. Breakouts above/below VHN with volume confirmation (>1.5x) indicate institutional participation. ATR(14) filter ensures volatility expansion. Works in bull markets (breakouts above VHN) and bear markets (breakdowns below VHN). Position size 0.25 manages drawdown. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3771_6h_donchian20_1d_vhn_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume profile VHN (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume profile high-volume node (VHN) - price level with max volume
    # Use 50 bins between 1d low and high
    nbins = 50
    vhn_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i < 1:  # Need at least 1 day of data
            continue
        # Create volume histogram for this 1d bar
        hist, bin_edges = np.histogram(
            [high_1d[i], low_1d[i], close_1d[i]],  # Simplified: use OHLC as proxy for price distribution
            bins=nbins,
            range=(low_1d[i], high_1d[i]),
            weights=[volume_1d[i], volume_1d[i], volume_1d[i]]  # Distribute volume across price range
        )
        # Find bin with maximum volume (VHN)
        if np.sum(hist) > 0:
            max_bin_idx = np.argmax(hist)
            vhn_1d[i] = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    
    # Align 1d VHN to 6h timeframe (shifted by 1 for completed 1d bar)
    vhn_1d_aligned = align_htf_to_ltf(prices, df_1d, vhn_1d)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility filter ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    atr_ratio = np.ones(n)
    atr_ratio[10:] = atr[10:] / atr_ma[10:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20, 14, 10)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vhn_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) AND volatility expansion (ATR > 1.2x MA)
        volume_spike = vol_ratio[i] > 1.5
        vol_expansion = atr_ratio[i] > 1.2
        
        if volume_spike and vol_expansion:
            # Long entry: Price breaks above Donchian upper band AND above 1d VHN (bullish breakout from value area)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > vhn_1d_aligned[i]):    # Above 1d volume high-volume node
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below 1d VHN (bearish breakdown from value area)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < vhn_1d_aligned[i]):    # Below 1d volume high-volume node
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals