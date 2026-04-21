#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 12h HTF trend filter (EMA34) and volume confirmation (>1.3x 20-period average) 
captures institutional interest levels. Choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend following) 
adapts to market conditions. ATR(14) trailing stop via signal=0 when price moves against position by 2.0*ATR. 
Target: 20-40 trades/year to minimize fee drag and work in both bull/bear markets via HTF alignment and regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA trend filter and Camarilla calculation)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h Camarilla levels (for primary timeframe alignment) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels: R1, S1 based on previous 12h bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_width = 1.1 * (high_12h - low_12h) / 12
    r1_12h = close_12h + camarilla_width
    s1_12h = close_12h - camarilla_width
    
    # Align Camarilla levels to 4h timeframe (use previous completed 12h bar)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * vol_ma
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period) for regime filter
    cp = pd.Series(high_4h - low_4h).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(cp / tr_sum) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) 
            or np.isnan(volume_threshold[i]) or np.isnan(atr[i]) 
            or np.isnan(ema_34_12h_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        if position == 0:
            # Regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
            in_range = chop[i] > 61.8
            in_trend = chop[i] < 38.2
            
            # Long conditions
            long_breakout = price > r1_12h_aligned[i]
            long_volume = volume_4h[i] > volume_threshold[i]
            long_trend = price > ema_34_12h_aligned[i]
            
            # Short conditions
            short_breakout = price < s1_12h_aligned[i]
            short_volume = volume_4h[i] > volume_threshold[i]
            short_trend = price < ema_34_12h_aligned[i]
            
            # Entry logic: adapt to regime
            if in_range:
                # In range: mean reversion at Camarilla levels
                if long_breakout and long_volume and not long_trend:  # Price above R1 but not strongly trending -> fade
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                elif short_breakout and short_volume and not short_trend:  # Price below S1 but not strongly trending -> fade
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            else:
                # In trend: breakout continuation
                if long_breakout and long_volume and long_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_breakout and short_volume and short_trend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Camarilla S1 (support broken)
            elif price < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Camarilla R1 (resistance broken)
            elif price > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_v2"
timeframe = "4h"
leverage = 1.0