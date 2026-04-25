#!/usr/bin/env python3
"""
4h Camarilla Pivot H3/L3 Breakout with 1d EMA50 Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance.
Breakouts above H3 or below L3 with 1d EMA50 trend alignment and volume confirmation
capture strong momentum moves in both bull and bear markets. The 1d EMA50 filter
avoids counter-trend trades, while volume spike ensures institutional participation.
Designed for 4h timeframe to target 20-50 trades/year by requiring confluence of
Camarilla breakout, 1d EMA50 trend, and volume confirmation, reducing overtrading.
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
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    # Range = high - low
    # H3 = close + (range * 1.1 / 4)
    # L3 = close - (range * 1.1 / 4)
    # We use previous day's data to avoid look-ahead
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    daily_range = df_1d['high'] - df_1d['low']
    camarilla_h3 = typical_price + (daily_range * 1.1 / 4)
    camarilla_l3 = typical_price - (daily_range * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and aligned data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
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
            long_entry = (curr_high > camarilla_h3_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Camarilla L3 AND bearish bias AND volume spike
            short_entry = (curr_low < camarilla_l3_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla L3 (mean reversion) OR loss of bullish bias
            if (curr_low < camarilla_l3_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 (mean reversion) OR loss of bearish bias
            if (curr_high > camarilla_h3_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0