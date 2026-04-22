# 6h_Pivots_DonchianBreakout_1dTrend_VolumeSpike
# Hypothesis: On 6H timeframe, combine daily pivot points (from 1D data) as support/resistance with
# 6H Donchian breakouts filtered by 1D EMA50 trend and volume spikes. This creates a strategy that
# captures breakouts from key daily levels in trending markets while avoiding chop.
# Works in bull/bear: In bull markets, buy breakouts above daily R1/R2 in uptrend; in bear markets,
# sell breakdowns below daily S1/S2 in downtrend. Volume spikes confirm institutional interest.
# Target: 15-30 trades/year per symbol (60-120 over 4 years) to stay well under 300 trade limit.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1D data for pivots, trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align daily pivot levels to 6H timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 1D EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6H Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Price and volume arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema50_1d_val = ema50_1d_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.8 * 20-period average volume (strict to reduce trades)
        vol_spike = vol > 1.8 * vol_ma
        
        # Trend filter: price above/below 1D EMA50
        uptrend = price > ema50_1d_val
        downtrend = price < ema50_1d_val
        
        if position == 0:
            # Long: price breaks above 6H Donchian high AND above daily R1 (resistance) 
            # AND in uptrend AND volume spike
            if price > donch_high_val and price > r1_val and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6H Donchian low AND below daily S1 (support) 
            # AND in downtrend AND volume spike
            elif price < donch_low_val and price < s1_val and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite Donchian level or trend changes
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low or trend turns down
                if price < donch_low_val or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high or trend turns up
                if price > donch_high_val or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Pivots_DonchianBreakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0