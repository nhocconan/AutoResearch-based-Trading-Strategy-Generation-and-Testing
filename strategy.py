#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop
Hypothesis: Camarilla pivot R1/S1 levels act as intraday support/resistance on 4h timeframe.
Breakout above R1 with volume confirmation and chop regime filter indicates bullish momentum.
Breakdown below S1 with volume confirmation and chop regime filter indicates bearish momentum.
ATR-based stoploss limits downside. Works in both bull/bear markets: regime filter adapts to conditions,
volume confirmation reduces false breakouts, and pivot levels provide structure in ranging markets.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h EMA34 for trend filter (HTF bias) ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla pivot levels from previous 4h bar
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # We need previous bar's high/low/close
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    
    # Set first value to NaN (no previous bar)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Volume confirmation: current volume > 1.5 * average volume (20-period)
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.5 * vol_ma)
    
    # Choppiness regime filter: CHOP(14) < 61.8 = trending (favor breakouts)
    # CHOP = 100 * log10(sum(TR,14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = max_high - min_low
    chop = np.where(denominator > 0, 
                    100 * np.log10(atr_sum / denominator) / np.log10(14), 
                    100)
    chop_filter = chop < 61.8  # Trending regime
    
    # ATR (14-period) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_34_12h_aligned[i]) 
            or np.isnan(volume_spike[i]) or np.isnan(chop_filter[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        if position == 0:
            # Long: Breakout above R1 + volume spike + trending regime + HTF bias long
            if (price > r1[i] and volume_spike[i] and chop_filter[i] 
                and close_4h[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Breakdown below S1 + volume spike + trending regime + HTF bias short
            elif (price < s1[i] and volume_spike[i] and chop_filter[i] 
                  and close_4h[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price returns below R1 or regime changes to choppy
            elif price < r1[i] or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price returns above S1 or regime changes to choppy
            elif price > s1[i] or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop"
timeframe = "4h"
leverage = 1.0