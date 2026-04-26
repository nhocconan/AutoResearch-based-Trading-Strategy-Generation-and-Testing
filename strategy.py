#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike_v1
Hypothesis: Donchian(20) breakout on 1d with 1-week EMA50 trend filter and volume confirmation (2x average) to reduce false breakouts. Uses 1w trend for better bull/bear discrimination - only long in strong uptrend, only short in strong downtrend. Designed for very low trade frequency (<25/year) to minimize fee drag in challenging 2025+ markets. Discrete position sizing (0.25) balances return vs risk. ATR-based stoploss (2.0x) manages risk. Focus on BTC/ETH as primary targets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for strong trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for strong trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss on 1d
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Donchian channels from previous 20 days
    # Highest high of previous 20 days (excluding current)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lowest low of previous 20 days (excluding current)
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1w EMA(50), volume MA, ATR, Donchian
    start_idx = max(50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1w_up = close_val > ema_50_1w_aligned[i]   # 1w strong uptrend
        trend_1w_down = close_val < ema_50_1w_aligned[i]  # 1w strong downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 1w trend up AND volume spike
            long_signal = (close_val > high_20[i]) and trend_1w_up and vol_spike
            
            # Short: price breaks below lower Donchian AND 1w trend down AND volume spike
            short_signal = (close_val < low_20[i]) and trend_1w_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_1w_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_1w_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0