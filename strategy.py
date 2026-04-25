#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike_v4
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation.
Only long when price breaks above R1 in 1d bull regime (close > EMA34), short when breaks below S1 in 1d bear regime (close < EMA34).
Volume confirmation requires volume > 2.0 * ATR(20) to avoid false breakouts.
Discrete sizing: 0.25. Target: 20-50 trades/year to minimize fee drag.
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
    
    # Get 1d data for trend regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend regime
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volume filter (using 4h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34) and ATR (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Calculate 4h Camarilla levels using previous bar's OHLC
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        rng = ph - pl
        
        # Camarilla R1 and S1 levels
        r1 = pc + (rng * 1.1 / 12)
        s1 = pc - (rng * 1.1 / 12)
        
        # Volume confirmation: current volume > 2.0 * ATR
        volume_spike = volume[i] > 2.0 * atr[i]
        
        # Determine 1d trend regime
        if close[i] > ema_34_1d_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_34_1d_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades (rare)
        
        if position == 0:
            # Long setup: price breaks above R1 AND volume spike AND bull regime
            long_setup = (close[i] > r1) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below S1 AND volume spike AND bear regime
            short_setup = (close[i] < s1) and volume_spike and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below S1 (reversal) OR regime turns bearish
            if (close[i] < s1) or (regime == 'bear'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above R1 (reversal) OR regime turns bullish
            if (close[i] > r1) or (regime == 'bull'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike_v4"
timeframe = "4h"
leverage = 1.0