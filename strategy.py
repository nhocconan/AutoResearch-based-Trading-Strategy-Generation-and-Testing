#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike filter and chop regime filter
# - Donchian breakout: price closes above 20-period high (long) or below 20-period low (short) on 4h
# - Volume confirmation: current 4h volume > 2.0x 20-period average to avoid false breakouts
# - Chop regime filter: only trade when Choppiness Index(14) < 38.2 (trending market) on 4h
# - ATR(14) trailing stop (2.5x) on 4h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within HARD MAX: 400 total
# - Works in bull markets via breakouts, works in bear via short breakouts + regime filter prevents whipsaws

name = "4h_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (not used in this strategy but keeping for potential extension)
    # df_1d = get_htf_data(prices, '1d')
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    # Pre-calculate indicators before loop
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) - measures trend vs range
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sumTR14 / (hh14 - ll14)) / log10(14)
    # Avoid division by zero and log of zero
    hl_range = hh_14 - ll_14
    chop_raw = np.where((hl_range > 0) & (tr_sum_14 > 0), 
                        100 * np.log10(tr_sum_14 / hl_range) / np.log10(14), 
                        50.0)  # default to neutral when invalid
    chop = chop_raw
    
    # ATR (14-period) for trailing stop
    tr_atr = tr  # reuse True Range from above
    atr_14 = pd.Series(tr_atr).rolling(window=14, min_periods=14).mean().values
    
    # Main loop
    for i in range(20, n):  # Start after warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop[i]) or 
            np.isnan(atr_14[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_price = close[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending markets (Chop < 38.2)
            if chop[i] < 38.2:
                # Volume confirmation: current volume > 2.0x 20-period average
                volume_spike = volume[i] > 2.0 * volume_ma_20[i]
                
                if volume_spike:
                    # Long breakout: close above 20-period high
                    if close_price > highest_20[i]:
                        position = 1
                        highest_since_entry = high[i]
                        signals[i] = 0.25
                    # Short breakout: close below 20-period low
                    elif close_price < lowest_20[i]:
                        position = -1
                        lowest_since_entry = low[i]
                        signals[i] = -0.25
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # ranging market - no breakout trades
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = close_price < highest_since_entry - 2.5 * atr_14[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, low[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = close_price > lowest_since_entry + 2.5 * atr_14[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals