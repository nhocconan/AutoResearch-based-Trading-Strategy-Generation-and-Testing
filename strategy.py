#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 1d chop regime filter (CHOP > 61.8 = range, mean reversion from R1/S1). 
In ranging markets (CHOP > 61.8), price tends to revert from extreme Camarilla levels (R1/S1). 
Volume confirmation reduces false breakouts. ATR-based stoploss manages risk. 
Target 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
Uses 4h primary timeframe with 1d HTF for chop regime calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Chop Regime (CHOP > 61.8 = ranging, mean revert) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR(14) and sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum_14 / (atr_14 * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Camarilla levels from previous 4h bar (for current bar breakout)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using previous bar's high/low/close to avoid look-ahead
    prev_high = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low = np.concatenate([[np.nan], low_4h[:-1]])
    prev_close = np.concatenate([[np.nan], close_4h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    # Calculate ATR for stoploss
    tr_4h1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr_4h2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr_4h3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] < entry_price - atr_multiplier * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] > entry_price + atr_multiplier * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        chop_val = chop_aligned[i]
        
        # Only trade in ranging markets (CHOP > 61.8)
        if chop_val > 61.8:
            if position == 0:
                # Long: price breaks above R1 + volume confirmation
                if price > r1[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: price breaks below S1 + volume confirmation
                elif price < s1[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            
            elif position == 1:
                # Check stoploss
                if price < entry_price - atr_multiplier * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit long: price breaks below S1 (mean reversion target) or volume fails
                elif price < s1[i] or not vol_ok:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Check stoploss
                if price > entry_price + atr_multiplier * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit short: price breaks above R1 (mean reversion target) or volume fails
                elif price > r1[i] or not vol_ok:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending markets, stay flat or follow existing position with reduced size
            if position != 0:
                # Check stoploss
                if position == 1 and price < entry_price - atr_multiplier * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and price > entry_price + atr_multiplier * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Reduce position in trending markets
                    signals[i] = 0.125 if position == 1 else -0.125
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0