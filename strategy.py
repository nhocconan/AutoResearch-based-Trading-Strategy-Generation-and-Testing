#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Uses Keltner Channel breakouts with 1d trend filter and volume spikes.
- Long: Price breaks above Keltner upper band + volume spike + price above 1d EMA34
- Short: Price breaks below Keltner lower band + volume spike + price below 1d EMA34
- Exit: Opposite Keltner band touch or loss of trend
Designed for low trade frequency by requiring confluence of volatility breakout, trend alignment, and volume confirmation.
Works in bull markets via breakouts and in bear markets via trend-filtered short opportunities.
"""

name = "4h_Keltner_Channel_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Keltner Channel (20-period EMA, 2x ATR) ---
    # EMA20
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR(20)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Keltner bands
    keltner_upper = ema_20 + 2 * atr_20
    keltner_lower = ema_20 - 2 * atr_20
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- 1d Trend Filter (EMA34 on 1d close) ---
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need EMA20 and ATR20)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Keltner upper + volume + above 1d EMA
            if (close[i] > keltner_upper[i] and 
                volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner lower + volume + below 1d EMA
            elif (close[i] < keltner_lower[i] and 
                  volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Keltner band touch or loss of trend
            if position == 1:
                # Exit long: price touches Keltner lower or loses trend
                if (close[i] < keltner_lower[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches Keltner upper or loses trend
                if (close[i] > keltner_upper[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals