#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation and ATR trailing stop
# Camarilla pivots from 12h provide intraday support/resistance levels with proven edge in ranging markets
# Volume confirmation (current 4h volume > 1.5x 20-period average) filters false breakouts
# ATR trailing stop (2.5x ATR) manages risk and adapts to volatility
# Designed for 4h timeframe targeting 25-50 trades/year (100-200 over 4 years)
# Works in bull/bear: price reacts to 12h pivot structure, volume confirms validity, ATR stop controls drawdown

name = "4h_12h_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla formulas for intraday trading
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    # H3 = Close + 1.1 * (High - Low) / 4
    # L3 = Close - 1.1 * (High - Low) / 4
    # H2 = Close + 1.1 * (High - Low) / 6
    # L2 = Close - 1.1 * (High - Low) / 6
    # H1 = Close + 1.1 * (High - Low) / 12
    # L1 = Close - 1.1 * (High - Low) / 12
    
    # Use previous 12h bar's high/low/close for current levels (no look-ahead)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    camarilla_h4 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 2
    camarilla_l4 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 2
    camarilla_h3 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 4
    camarilla_l3 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 4
    camarilla_h2 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 6
    camarilla_l2 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 6
    camarilla_h1 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 12
    camarilla_l1 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 12
    
    # Align 12h Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l1)
    
    # Pre-compute ATR(14) for 4h timeframe
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
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
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
            # Short at H4 resistance, Long at L4 support
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