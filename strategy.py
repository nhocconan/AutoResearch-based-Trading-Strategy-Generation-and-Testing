#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels with volume confirmation and ATR trailing stop
# - Uses 1w HTF for Camarilla pivot calculation (based on completed weekly candles)
# - Long when price breaks above weekly R4 level with volume > 2.0x 20-period average
# - Short when price breaks below weekly S4 level with volume > 2.0x 20-period average
# - ATR(14) trailing stop: exit long at 2.5x ATR below highest high since entry, exit short at 2.5x ATR above lowest low since entry
# - Fixed position size 0.25 to control drawdown
# - Weekly pivots adapt to volatility and provide significant support/resistance levels
# - Volume confirmation filters false breakouts
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1w_camarilla_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels (based on previous week's OHLC)
    # Camarilla formulas:
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1 / 2)
    # S4 = C - ((H - L) * 1.1 / 2)
    # We use the previous week's values to avoid look-ahead
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    camarilla_r4_1w = close_1w + ((high_1w - low_1w) * 1.1 / 2.0)
    camarilla_s4_1w = close_1w - ((high_1w - low_1w) * 1.1 / 2.0)
    
    # Shift by 1 to use previous week's levels (avoid look-ahead)
    pivot_1w_shifted = np.roll(pivot_1w, 1)
    camarilla_r4_1w_shifted = np.roll(camarilla_r4_1w, 1)
    camarilla_s4_1w_shifted = np.roll(camarilla_s4_1w, 1)
    pivot_1w_shifted[0] = np.nan
    camarilla_r4_1w_shifted[0] = np.nan
    camarilla_s4_1w_shifted[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1w bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w_shifted)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w_shifted)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR (14-period) for stoploss
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
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or
            vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation
            if volume_confirmed:
                # Long entry: price breaks above weekly R4 level
                if close[i] > camarilla_r4_aligned[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price breaks below weekly S4 level
                elif close[i] < camarilla_s4_aligned[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals