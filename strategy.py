#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h HTF Camarilla pivot levels (H3/L3) with volume confirmation and ATR trailing stop
# - Uses 12h HTF for Camarilla pivot calculation (based on completed 12h candles)
# - Long when price breaks above H3 level with volume > 1.5x 20-period average
# - Short when price breaks below L3 level with volume > 1.5x 20-period average
# - ATR(14) trailing stop: exit long at 2.0x ATR below highest high since entry, exit short at 2.0x ATR above lowest low since entry
# - Fixed position size 0.25 to control drawdown
# - Camarilla pivots work well in ranging markets (common in 2025 BTC/ETH bear/range) and capture breakouts in trending markets
# - Volume confirmation filters false breakouts, ATR stop manages risk
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)

name = "4h_12h_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Typical price = (high + low + close) / 3
    typical_price = (high_12h + low_12h + close_12h) / 3.0
    # Range = high - low
    price_range = high_12h - low_12h
    
    # Camarilla levels: H3 = close + (range * 1.1/4), L3 = close - (range * 1.1/4)
    # Using previous bar's values to avoid look-ahead
    camarilla_high = close_12h + (price_range * 1.1 / 4.0)
    camarilla_low = close_12h - (price_range * 1.1 / 4.0)
    
    # Shift by 1 to use previous bar's levels (completed bar)
    camarilla_high = np.roll(camarilla_high, 1)
    camarilla_low = np.roll(camarilla_low, 1)
    camarilla_high[0] = np.nan
    camarilla_low[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe (wait for completed 12h bar)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Pre-compute volume confirmation (20-period average for 4h)
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
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or
            vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
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
            # Entry logic: Camarilla breakout with volume confirmation
            if volume_confirmed:
                # Long entry: price breaks above Camarilla H3 level
                if close[i] > camarilla_high_aligned[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price breaks below Camarilla L3 level
                elif close[i] < camarilla_low_aligned[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals