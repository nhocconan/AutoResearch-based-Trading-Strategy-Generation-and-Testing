#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and ATR trailing stop
# Camarilla pivots from 1d provide intraday support/resistance levels with proven mean-reversion edge
# Volume confirmation (current 12h volume > 2.0x 20-period average) filters false breakouts
# ATR trailing stop (2.5x ATR) manages risk and adapts to volatility
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: price reacts to 1d Camarilla structure, volume confirms validity, ATR stop controls drawdown

name = "12h_1d_camarilla_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot point = (High + Low + Close) / 3
    # Range = High - Low
    # L3 = Pivot - Range * 1.1 / 4
    # L4 = Pivot - Range * 1.1 / 2
    # H3 = Pivot + Range * 1.1 / 4
    # H4 = Pivot + Range * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    h3 = pp + rng * 1.1 / 4.0
    h4 = pp + rng * 1.1 / 2.0
    l3 = pp - rng * 1.1 / 4.0
    l4 = pp - rng * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute ATR(14) for 12h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average 12h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 2.5x ATR from highest
            if close[i] < highest_since_long - 2.5 * atr[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 2.5x ATR from lowest
            if close[i] > lowest_since_short + 2.5 * atr[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion trading at Camarilla extremes with volume confirmation
            # Short at H4, Long at L4
            if volume_confirmed:
                if close[i] > h4_aligned[i]:
                    position = -1
                    lowest_since_short = close[i]
                    signals[i] = -0.25
                elif close[i] < l4_aligned[i]:
                    position = 1
                    highest_since_long = close[i]
                    signals[i] = 0.25
    
    return signals