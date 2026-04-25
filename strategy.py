#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with 12h EMA50 Trend Filter and Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum. When aligned with 12h EMA50 trend and confirmed by volume spikes,
this strategy filters false breakouts and trades with institutional flow. Designed for 4h timeframe to target 20-50 trades/year
(80-200 over 4 years) by requiring confluence of Donchian breakout, 12h EMA trend, and volume confirmation. Works in bull/bear
regimes via trend filter and volume requirement for validity. Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 2.5 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    # Donchian(20) channels on primary timeframe (4h)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20), EMA50, volume MA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 12h EMA50
        bullish_bias = curr_close > ema_12h_aligned[i]
        bearish_bias = curr_close < ema_12h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Donchian breakout + trend + volume spike
            # Long: price breaks above Donchian(20) high AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_high[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian(20) low AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_low[i]) and bearish_bias and vol_spike
            
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
            # Exit: price falls below Donchian(20) low (mean reversion) OR loss of bullish bias
            if (curr_low < donchian_low[i]) or (curr_close < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian(20) high (mean reversion) OR loss of bearish bias
            if (curr_high > donchian_high[i]) or (curr_close > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0