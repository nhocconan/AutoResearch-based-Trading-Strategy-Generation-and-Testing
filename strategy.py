#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with Daily EMA Trend and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance. 
Breakouts above H3 or below L3 with daily EMA trend alignment and volume confirmation 
capture momentum moves in both bull and bear markets. Discrete sizing (0.0, ±0.25) 
minimizes fee churn. Target: 20-50 trades/year on 4h.
"""

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
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    phigh = df_1d['high'].shift(1).values
    plow = df_1d['low'].shift(1).values
    pclose = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    range_ = phigh - plow
    H3 = pclose + (range_ * 1.1 / 4)
    L3 = pclose - (range_ * 1.1 / 4)
    H4 = pclose + (range_ * 1.1 / 2)
    L4 = pclose - (range_ * 1.1 / 2)
    
    # Align to 4h timeframe (wait for daily close)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Daily trend: price above/below EMA34
        trend_up = ema_34_aligned[i] > 0 and curr_close > ema_34_aligned[i]
        trend_down = ema_34_aligned[i] > 0 and curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND uptrend
            long_entry = (curr_close > H3_aligned[i]) and vol_spike and trend_up
            # Short: price breaks below L3 AND volume spike AND downtrend
            short_entry = (curr_close < L3_aligned[i]) and vol_spike and trend_down
            
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
            # Exit: price falls below L3 (reversal) OR loss of trend (price < EMA34)
            if (curr_close < L3_aligned[i]) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (reversal) OR loss of trend (price > EMA34)
            if (curr_close > H3_aligned[i]) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0