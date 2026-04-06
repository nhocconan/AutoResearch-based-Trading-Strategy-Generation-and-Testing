#!/usr/bin/env python3
"""
6h Weekly Pivot Point + Volume Confirmation Strategy
Hypothesis: Weekly pivot points (PP, R1, S1, R2, S2) act as strong support/resistance levels. 
Price rejecting S1/R1 with volume confirmation indicates institutional interest. 
In bull market: buy near S1/S2 with bullish weekly candle. 
In bear market: sell near R1/R2 with bearish weekly candle.
Uses weekly pivot from previous week (no look-ahead) and 6h volume spike for confirmation.
Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weeklypivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get weekly OHLC for pivot calculation (previous week's data)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Calculate weekly pivot points: PP = (H + L + C)/3
    # R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H - L), S2 = PP - (H - L)
    pp = np.full(len(weekly_close), np.nan)
    r1 = np.full(len(weekly_close), np.nan)
    s1 = np.full(len(weekly_close), np.nan)
    r2 = np.full(len(weekly_close), np.nan)
    s2 = np.full(len(weekly_close), np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            pp[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            r1[i] = 2 * pp[i] - weekly_low[i]
            s1[i] = 2 * pp[i] - weekly_high[i]
            r2[i] = pp[i] + (weekly_high[i] - weekly_low[i])
            s2[i] = pp[i] - (weekly_high[i] - weekly_low[i])
    
    # Align weekly pivot levels to 6h timeframe (use previous week's values)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Weekly candle direction (bullish/bearish) from previous week
    weekly_bullish = np.where(weekly_close > weekly_open, 1, -1)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish)
    
    # 20-period average volume on 6h data
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current volume > 1.8x average volume
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Price levels
        pp_level = pp_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price falls below S1 OR weekly turns bearish
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s1_level or
                weekly_bullish_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above R1 OR weekly turns bullish
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r1_level or
                weekly_bullish_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Long setup: price near S1/S2 with bullish weekly candle + volume
                near_s1 = abs(close[i] - s1_level) < (pp_level - s1_level) * 0.3
                near_s2 = abs(close[i] - s2_level) < (pp_level - s2_level) * 0.3
                near_support = near_s1 or near_s2
                
                # Short setup: price near R1/R2 with bearish weekly candle + volume
                near_r1 = abs(close[i] - r1_level) < (r1_level - pp_level) * 0.3
                near_r2 = abs(close[i] - r2_level) < (r2_level - pp_level) * 0.3
                near_resistance = near_r1 or near_r2
                
                # Long: near support with bullish weekly + volume
                if near_support and weekly_bullish_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: near resistance with bearish weekly + volume
                elif near_resistance and weekly_bullish_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals