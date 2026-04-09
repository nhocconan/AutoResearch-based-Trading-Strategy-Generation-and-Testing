#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and ATR trailing stop
# - Uses 1d HTF to calculate Camarilla pivot levels (H3, L3) from prior completed day
# - Long when price closes above H3 with volume > 2.0x 30-period average volume
# - Short when price closes below L3 with volume > 2.0x 30-period average volume
# - ATR(20) trailing stop: exit long at 3.0x ATR below highest high since entry
# - ATR(20) trailing stop: exit short at 3.0x ATR above lowest low since entry
# - Fixed position size 0.25 to control drawdown
# - Volume filter reduces false breakouts, ATR stop manages risk
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Works in bull markets via breakouts, in bear via short opportunities and tight stops

name = "12h_1d_camarilla_breakout_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior day's Camarilla levels (H3, L3 for breakout)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    high_low_diff = high_1d - low_1d
    H3 = close_1d + 1.0 * high_low_diff
    L3 = close_1d - 1.0 * high_low_diff
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Pre-compute volume confirmation (30-period average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Pre-compute ATR (20-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_ma_30[i]) or np.isnan(atr[i]) or
            vol_ma_30[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_30[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 3.0x ATR from highest high
            if close[i] < highest_high_since_entry - 3.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 3.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 3.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout + volume confirmation
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