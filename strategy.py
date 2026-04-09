#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels with volume confirmation and ATR trailing stop
# Camarilla levels from 1w provide strong support/resistance structure proven to work across market regimes
# Volume confirmation (current 1d volume > 2.0x 20-period average) filters false breakouts
# ATR trailing stop (2.5x ATR) manages risk and adapts to volatility
# Designed for 1d timeframe targeting 15-25 trades/year (60-100 over 4 years)
# Works in bull/bear: price reacts to 1w structure, volume confirms validity, ATR stop controls drawdown

name = "1d_1w_camarilla_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (based on prior week)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L4 = C - Range * 1.1/2
    # H4 = C + Range * 1.1/2
    # L3 = C - Range * 1.1/4
    # H3 = C + Range * 1.1/4
    # L2 = C - Range * 1.1/6
    # H2 = C + Range * 1.1/6
    # L1 = C - Range * 1.1/12
    # H1 = C + Range * 1.1/12
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w
    
    h4 = close_1w + rng * 1.1 / 2.0
    l4 = close_1w - rng * 1.1 / 2.0
    h3 = close_1w + rng * 1.1 / 4.0
    l3 = close_1w - rng * 1.1 / 4.0
    h2 = close_1w + rng * 1.1 / 6.0
    l2 = close_1w - rng * 1.1 / 6.0
    h1 = close_1w + rng * 1.1 / 12.0
    l1 = close_1w - rng * 1.1 / 12.0
    
    # Align 1w Camarilla levels to 1d timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Pre-compute ATR(14) for 1d timeframe
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
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x average 1d volume
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
            # Camarilla breakout trading with volume confirmation
            # Long on H4 breakout, Short on L4 breakdown
            if volume_confirmed:
                if close[i] > h4_aligned[i]:
                    position = 1
                    highest_since_long = close[i]
                    signals[i] = 0.25
                elif close[i] < l4_aligned[i]:
                    position = -1
                    lowest_since_short = close[i]
                    signals[i] = -0.25
    
    return signals