#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d HTF volume confirmation and ATR trailing stop
# - Uses 1d HTF to calculate Camarilla pivot levels (H3, L3) from prior day
# - Long when 6h price closes above H3 with 1d volume > 1.5x 20-period average
# - Short when 6h price closes below L3 with 1d volume > 1.5x 20-period average
# - ATR(14) trailing stop: exit long at 2.0x ATR below highest high since entry
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)
# - Uses 1d volume for confirmation to reduce false breakouts and align with HTF structure

name = "6h_1d_camarilla_breakout_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume_6h = prices['volume'].values  # 6h volume for entry timing
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate prior day's Camarilla levels
    high_low_diff = high_1d - low_1d
    H3 = close_1d + 1.0 * high_low_diff
    L3 = close_1d - 1.0 * high_low_diff
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute ATR (14-period) for stoploss using 6h data
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr[i]) or
            vol_ma_20_1d_aligned[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x average (moderate filter)
        volume_confirmed = volume_1d[-1] > 1.5 * vol_ma_20_1d_aligned[i] if len(volume_1d) > 0 else False
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout + 1d volume confirmation
            if volume_confirmed:
                # Long entry: price closes above H3
                if close[i] > H3_aligned[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price closes below L3
                elif close[i] < L3_aligned[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals