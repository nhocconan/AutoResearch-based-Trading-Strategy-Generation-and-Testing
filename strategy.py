#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v2
# Strategy: 4h Camarilla pivot breakout with volume confirmation and daily volatility filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels from daily chart act as key support/resistance. Breakouts with volume capture institutional flow.
# Daily volatility filter avoids choppy markets. Designed for ~20-40 trades/year to minimize fee drag.
# Works in bull markets via long breakouts above H4 and in bear markets via short breakdowns below L4.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Formula: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero in range calculation
    prev_range = prev_high - prev_low
    # Where range is zero, use a small epsilon to avoid invalid levels
    prev_range = np.where(prev_range == 0, 0.0001, prev_range)
    
    H4 = prev_close + prev_range * 1.1 / 2
    L4 = prev_close - prev_range * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Daily volatility filter: use ATR(10) to avoid ranging markets
    # Calculate ATR on 1d timeframe
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(atr_10_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Volatility filter: avoid trading when ATR is too low (choppy market)
        # Use 50-period average of ATR as threshold
        if i >= 50:
            atr_avg_50 = np.nanmean(atr_10_aligned[i-50:i]) if not np.isnan(np.nanmean(atr_10_aligned[i-50:i])) else 0
            vol_filter = atr_10_aligned[i] > 0.5 * atr_avg_50  # Only trade when volatility is above half of recent average
        else:
            vol_filter = True  # Not enough data, allow trade
        
        # Breakout signals
        breakout_up = high[i] > H4_aligned[i]
        breakdown_down = low[i] < L4_aligned[i]
        
        # Entry conditions
        # Long: Breakout above H4 with volume confirmation and volatility filter
        if breakout_up and vol_confirm and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L4 with volume confirmation and volatility filter
        elif breakdown_down and vol_confirm and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout (breakdown for long, breakout for short)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals