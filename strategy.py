#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla pivot with weekly volume confirmation and ATR volatility filter
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Trading reversals at these levels
# with weekly volume confirmation avoids false signals, and volatility filter prevents whipsaws in low-volatility regimes.
# Works in bull via bounces at support/resistance, in bear via mean-reversion at extremes.
# Target: 15-35 trades/year to minimize fee drag.
name = "12h_camarilla_pivot_1w_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (based on previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (using previous week's OHLC)
    # Camarilla formulas: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # H2 = Close + 0.55 * (High - Low)
    # H1 = Close + 0.275 * (High - Low)
    # L1 = Close - 0.275 * (High - Low)
    # L2 = Close - 0.55 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # We use H3/L3 for entry and H4/L4 for stop
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    camarilla_width = 1.1 * (wk_high - wk_low)
    h3 = wk_close + camarilla_width
    l3 = wk_close - camarilla_width
    h4 = wk_close + 1.5 * (wk_high - wk_low)
    l4 = wk_close - 1.5 * (wk_high - wk_low)
    
    # Align weekly levels to 12h timeframe (shifted by 1 week for lookback)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Get daily volume for confirmation (using previous day's average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after sufficient warmup
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low volatility chop)
        vol_filter = atr[i] > atr_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches L3 (mean reversion target) OR volatility drops
            if close[i] <= l3_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches H3 (mean reversion target) OR volatility drops
            if close[i] >= h3_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry at extreme levels
            # Enter long: price touches L4 (extreme support) + volume confirmation + volatility filter
            if close[i] <= l4_aligned[i] and vol_confirm and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H4 (extreme resistance) + volume confirmation + volatility filter
            elif close[i] >= h4_aligned[i] and vol_confirm and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals