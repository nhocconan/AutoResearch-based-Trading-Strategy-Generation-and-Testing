#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla Pivot with 1d Trend and Volume Confirmation
# Hypothesis: Camarilla pivot levels act as strong support/resistance on 12h timeframe.
# Trading reversals from these levels with 1d trend filter and volume confirmation
# provides edge in both bull and bear markets. Targets 15-35 trades/year.

name = "12h_camarilla_pivot_1d_trend_volume_v1"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's OHLC for Camarilla pivot calculation
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Get previous day's high, low, close
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    prev_close = df_1d_ohlc['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    # Camarilla formulas: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.6*(high-low)
    # H1 = close + 0.3*(high-low)
    # L1 = close - 0.3*(high-low)
    # L2 = close - 0.6*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    
    range_hl = prev_high - prev_low
    
    camarilla_h4 = prev_close + 1.5 * range_hl
    camarilla_h3 = prev_close + 1.1 * range_hl
    camarilla_h2 = prev_close + 0.6 * range_hl
    camarilla_h1 = prev_close + 0.3 * range_hl
    camarilla_l1 = prev_close - 0.3 * range_hl
    camarilla_l2 = prev_close - 0.6 * range_hl
    camarilla_l3 = prev_close - 1.1 * range_hl
    camarilla_l4 = prev_close - 1.5 * range_hl
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    h4_12h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h4)
    h3_12h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h3)
    h2_12h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h2)
    h1_12h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h1)
    l1_12h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l1)
    l2_12h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l2)
    l3_12h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l3)
    l4_12h = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l4)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(ema50_12h[i]) or 
            np.isnan(h4_12h[i]) or np.isnan(h3_12h[i]) or np.isnan(h2_12h[i]) or np.isnan(h1_12h[i]) or
            np.isnan(l1_12h[i]) or np.isnan(l2_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 or trend turns down
            if close[i] < l3_12h[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above H3 or trend turns up
            if close[i] > h3_12h[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price bounces from L3/L4 with volume confirmation + uptrend
            if (close[i] > l3_12h[i] and 
                close[i] < l4_12h[i] * 1.001 and  # Allow small buffer for touching L4
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price rejects from H3/H4 with volume confirmation + downtrend
            elif (close[i] < h3_12h[i] and 
                  close[i] > h4_12h[i] * 0.999 and  # Allow small buffer for touching H4
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals