#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and 1w regime filter
# - Long when price breaks above R4 with volume spike and 1w close > EMA50 (bull regime)
# - Short when price breaks below S4 with volume spike and 1w close < EMA50 (bear regime)
# - Exit when price retouches the pivot point (mean reversion to equilibrium)
# - Uses discrete sizing (0.25) to minimize fee churn
# - Designed for 6h: targets 15-35 trades/year (60-140 total over 4 years)
# - Works in bull/bear: 1w EMA50 filter ensures we only trade breakouts in direction of weekly trend

name = "6h_1d_1w_camarilla_breakout_pivot_exit_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1w EMA50 for regime filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 6h ATR(14) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    pivot = 0.0
    r4 = 0.0
    s4 = 0.0
    
    for i in range(20, n):
        # Need prior day's OHLC for Camarilla calculation
        if i < 24:  # Need at least 1 day of 6h bars (24 bars)
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from prior 1d candle
        # Prior 1d candle ended at index i-1 (since we're at bar i)
        # Convert 6h index to 1d index: each 1d = 4 * 6h bars
        idx_1d = (i // 4) - 1  # Prior completed 1d candle
        if idx_1d < 0 or idx_1d >= len(df_1d):
            signals[i] = 0.0
            continue
            
        # Get prior 1d OHLC
        h_1d = df_1d['high'].iloc[idx_1d]
        l_1d = df_1d['low'].iloc[idx_1d]
        c_1d = df_1d['close'].iloc[idx_1d]
        
        # Camarilla levels
        range_1d = h_1d - l_1d
        pivot = (h_1d + l_1d + c_1d) / 3
        r4 = pivot + (range_1d * 1.1 / 2)
        s4 = pivot - (range_1d * 1.1 / 2)
        
        if position == 1:  # Long position
            # Exit: price retouches pivot (mean reversion) or ATR stop
            if prices['close'].iloc[i] <= pivot or prices['close'].iloc[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retouches pivot (mean reversion) or ATR stop
            if prices['close'].iloc[i] >= pivot or prices['close'].iloc[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume and regime filters
            if vol_spike[i]:
                # Long breakout above R4 in bull regime (1w close > EMA50)
                if prices['close'].iloc[i] > r4 and close_1w[i] > ema_50_1w_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short breakout below S4 in bear regime (1w close < EMA50)
                elif prices['close'].iloc[i] < s4 and close_1w[i] < ema_50_1w_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals