#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_4hEMA50_Trend_VolumeSpike_v1
Hypothesis: 4h Donchian(20) breakout with 4h EMA50 trend filter and volume confirmation (2.0x average) to capture medium-term swings in both bull and bear markets. Uses discrete position sizing (0.25) and ATR-based stoploss (2.5x) for risk management. Targets 20-50 trades/year to minimize fee drag while maintaining edge. The EMA50 trend filter ensures we only trade in the direction of the 4h trend, reducing false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), EMA(50), volume MA, ATR
    start_idx = max(20, 50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
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
        trend_up = close_val > ema_50[i]   # 4h uptrend
        trend_down = close_val < ema_50[i]  # 4h downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 4h trend up AND volume spike
            long_signal = (close_val > highest_high[i]) and trend_up and vol_spike
            
            # Short: price breaks below lower Donchian AND 4h trend down AND volume spike
            short_signal = (close_val < lowest_low[i]) and trend_down and vol_spike
            
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
            if (not trend_up) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_down) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0