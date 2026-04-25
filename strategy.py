#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with 1d EMA50 Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. Breakouts beyond these levels
with 1d EMA50 trend alignment and volume confirmation capture institutional momentum.
Works in both bull (long R3 breakouts) and bear (short S3 breakouts) regimes by requiring confluence.
Target: 20-50 trades/year via strict entry conditions (Camarilla breakout + trend + volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H+L+C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    camarilla_h4 = typical_price + (range_hl * 1.1/2)
    camarilla_l4 = typical_price - (range_hl * 1.1/2)
    camarilla_h3 = typical_price + (range_hl * 1.1/4)
    camarilla_l3 = typical_price - (range_hl * 1.1/4)
    camarilla_h2 = typical_price + (range_hl * 1.1/6)
    camarilla_l2 = typical_price - (range_hl * 1.1/6)
    camarilla_h1 = typical_price + (range_hl * 1.1/12)
    camarilla_l1 = typical_price - (range_hl * 1.1/12)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and aligned HTF data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Camarilla breakout + trend + volume
            # Long: price breaks above Camarilla H3 AND bullish bias AND volume spike
            long_entry = (curr_high > h3_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Camarilla L3 AND bearish bias AND volume spike
            short_entry = (curr_low < l3_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla L3 (mean reversion) OR loss of bullish bias
            if (curr_low < l3_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 (mean reversion) OR loss of bearish bias
            if (curr_high > h3_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0