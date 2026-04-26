#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: On 12h timeframe, Donchian channel (20) breakouts filtered by 1d trend (price > 1d EMA50) and volume spike (>1.5x 20-period average) capture medium-term momentum with controlled trade frequency. Uses ATR-based stoploss (signal=0 when price moves against position by 2.5x ATR). Discrete sizing (±0.25) targets 12-37 trades/year. Works in bull/bear markets by only trading in direction of higher-timeframe trend.
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 12h
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >1.5x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR (20) for stoploss calculation
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0  # first bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of Donchian (20), volume MA (20), ATR (20) + alignment buffer
    start_idx = 20 + 4  # +4 to ensure 1d bar completion (12h -> 1d: 2 bars per day, but add buffer)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donch_high = period20_high[i]
        donch_low = period20_low[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: Donchian breakout in direction of 1d trend WITH volume spike
        long_entry = (close_val > donch_high) and bullish_1d and vol_spike
        short_entry = (close_val < donch_low) and bearish_1d and vol_spike
        
        # Stoploss conditions: close-based ATR stop
        long_stop = False
        short_stop = False
        if position == 1 and entry_price > 0:
            long_stop = close_val < (entry_price - 2.5 * atr_val)
        if position == -1 and entry_price > 0:
            short_stop = close_val > (entry_price + 2.5 * atr_val)
        
        # Exit on stoploss or trend reversal
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif position == 1 and (long_stop or not bullish_1d):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        elif position == -1 and (short_stop or not bearish_1d):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0