#!/usr/bin/env python3
"""
6h_Keltner_Donchian_Squeeze_Breakout_12hTrend_VolumeSpike
Hypothesis: 6-hour volatility squeeze (Keltner width < Donchian width) followed by breakout in direction of 12h EMA50 trend with volume confirmation. Works in bull/bear by using 12h trend filter and breakouts from low volatility periods. Targets 12-30 trades/year to minimize fee drag while capturing explosive moves after consolidation.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(20) for Keltner channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA(20) for Keltner middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel width: 2 * ATR(20)
    keltner_width = 2.0 * atr
    
    # Donchian Channel width: (20-period high - 20-period low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_width = high_20 - low_20
    
    # Volatility squeeze: Keltner width < Donchian width (low volatility)
    volatility_squeeze = keltner_width < donchian_width
    
    # Donchian breakout levels
    donchian_high = high_20
    donchian_low = low_20
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 12h EMA(50), ATR(20), EMA(20), Donchian(20)
    start_idx = max(50, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_12h_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_12h_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        squeeze = volatility_squeeze[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Donchian breakout above AND 12h trend up AND volatility squeeze AND volume spike
            long_signal = (close_val > donchian_high[i]) and trend_12h_up and squeeze and vol_spike
            
            # Short: Donchian breakdown below AND 12h trend down AND volatility squeeze AND volume spike
            short_signal = (close_val < donchian_low[i]) and trend_12h_down and squeeze and vol_spike
            
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
            # Exit: 12h trend flips down OR price retracement to EMA(20)
            if (not trend_12h_up) or (close_val < ema_20[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 12h trend flips up OR price retracement to EMA(20)
            if (not trend_12h_down) or (close_val > ema_20[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Keltner_Donchian_Squeeze_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0