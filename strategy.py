#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator trend filter + 1d RSI mean-reversion with volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction on 4h, enters on 1d RSI extremes
# only when price is aligned with the trend. Volume spike confirms momentum. Designed to work in
# both bull (trend following) and bear (mean reversion within trend) markets.
# Target: 20-50 total trades over 4 years (5-12/year) with size 0.25.

name = "4h_WilliamsAlligator_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 4h: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    median_price_4h = (df_4h['high'] + df_4h['low']) / 2
    jaw = median_price_4h.rolling(window=13, min_periods=13).mean()
    teeth = median_price_4h.rolling(window=8, min_periods=8).mean()
    lips = median_price_4h.rolling(window=5, min_periods=5).mean()
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips_vals)
    
    # Trend: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
    uptrend = lips_aligned > teeth_aligned
    downtrend = lips_aligned < teeth_aligned
    
    # 1-day RSI for mean-reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_vals = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_vals)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: uptrend + RSI oversold (<30) + volume spike
            if uptrend[i] and (rsi_aligned[i] < 30) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + RSI overbought (>70) + volume spike
            elif downtrend[i] and (rsi_aligned[i] > 70) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend breaks (lips < teeth) OR RSI overbought (>70)
            if (not uptrend[i]) or (rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend breaks (lips > teeth) OR RSI oversold (<30)
            if (not downtrend[i]) or (rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals