#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) on 6h timeframe, with breakouts confirmed by 
1d trend (price > EMA34) and volume spikes (>2x 48-period average), capture momentum 
continuation in both bull and bear markets. Fades at R3/S3 in ranging markets, breaks 
out at R4/S4 in trending markets. Position size 0.25 limits risk. Targets 15-25 
trades/year to minimize fee drag.
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 48-period average (2 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # Camarilla pivot levels (based on previous day's range)
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # We use 1d OHLC to calculate daily pivot levels
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero or NaN
    valid = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
    
    # Camarilla levels
    R3 = prev_close + 1.0 * (prev_high - prev_low)
    S3 = prev_close - 1.0 * (prev_high - prev_low)
    R4 = prev_close + 1.5 * (prev_high - prev_low)
    S4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align pivot levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(48, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above R3 with volume confirmation and uptrend
            # In ranging markets: fade at R3/S3 (mean reversion)
            # In trending markets: breakout continuation at R4/S4
            if (close[i] > R3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                # Check if it's a strong breakout (above R4) or just touching R3
                if close[i] > R4_aligned[i]:
                    # Strong breakout - go long
                    signals[i] = 0.25
                    position = 1
                else:
                    # Weak breakout at R3 - fade (go short) in ranging markets
                    # But only if price is below EMA (downtrend) for confirmation
                    if close[i] < ema34_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                    else:
                        # Uptrend but weak breakout - wait for stronger signal
                        signals[i] = 0.0
            # SHORT: Breakdown below S3 with volume confirmation and downtrend
            elif (close[i] < S3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                # Check if it's a strong breakdown (below S4) or just touching S3
                if close[i] < S4_aligned[i]:
                    # Strong breakdown - go short
                    signals[i] = -0.25
                    position = -1
                else:
                    # Weak breakdown at S3 - fade (go long) in ranging markets
                    # But only if price is above EMA (uptrend) for confirmation
                    if close[i] > ema34_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    else:
                        # Downtrend but weak breakdown - wait for stronger signal
                        signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend reverses
            if (close[i] < R3_aligned[i]) or \
               (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend reverses
            if (close[i] > S3_aligned[i]) or \
               (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals