#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot levels + volume confirmation + session filter (08-20 UTC)
# Camarilla pivots provide intraday support/resistance levels based on prior 4h range
# Long when price breaks above H3 level with volume confirmation during active session
# Short when price breaks below L3 level with volume confirmation during active session
# Uses discrete position sizing 0.20 to target ~15-30 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, session filter avoids low-liquidity periods

name = "1h_4h_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (based on prior 4h bar)
    # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low)
    # L3 = close - 1.25*(high-low), L4 = close - 1.5*(high-low)
    camarilla_h4 = close_4h + 1.5 * (high_4h - low_4h)
    camarilla_h3 = close_4h + 1.25 * (high_4h - low_4h)
    camarilla_l3 = close_4h - 1.25 * (high_4h - low_4h)
    camarilla_l4 = close_4h - 1.5 * (high_4h - low_4h)
    
    # Align 4h indicators to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.8x average 1h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if price falls below L3 level
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3 level
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Breakout strategy: enter on Camarilla breakout with volume confirmation
            if close[i] > camarilla_h3_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.20
            elif close[i] < camarilla_l3_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.20
    
    return signals