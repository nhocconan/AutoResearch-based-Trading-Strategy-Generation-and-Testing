#!/usr/bin/env python3
"""
4h Camarilla Pivot H3/L3 Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Breakouts above H3 (bullish) or below L3 (bearish)
capture momentum when aligned with 1d EMA34 trend and confirmed by volume spikes. Designed for 4h timeframe
to target 20-50 trades/year by requiring confluence of Camarilla breakout, 1d EMA34 trend, and volume confirmation,
reducing overtrading and fee drag. Works in bull (long H3 breakouts) and bear (short L3 breakouts) regimes.
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    # Range = high - low
    typical_price = (high + low + close) / 3.0
    daily_range = high - low
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_typical = pd.Series(typical_price).shift(1).values
    prev_range = pd.Series(daily_range).shift(1).values
    
    # Camarilla levels based on previous day
    # H3 = typical + (range * 1.1/4) = typical + range * 0.275
    # L3 = typical - (range * 1.1/4) = typical - range * 0.275
    camarilla_h3 = prev_typical + (prev_range * 0.275)
    camarilla_l3 = prev_typical - (prev_range * 0.275)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla calculation and EMA34
    start_idx = max(20, 34)  # Need previous day data + EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Camarilla breakout + trend + volume
            # Long: price breaks above Camarilla H3 AND bullish bias AND volume spike
            long_entry = (curr_high > camarilla_h3[i]) and bullish_bias and vol_spike
            # Short: price breaks below Camarilla L3 AND bearish bias AND volume spike
            short_entry = (curr_low < camarilla_l3[i]) and bearish_bias and vol_spike
            
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
            if (curr_low < camarilla_l3[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 (mean reversion) OR loss of bearish bias
            if (curr_high > camarilla_h3[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0