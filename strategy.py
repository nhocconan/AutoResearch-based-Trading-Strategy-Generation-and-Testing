#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR trend filter and volume spike confirmation.
- Primary timeframe: 12h for lower trade frequency (~20-50 trades/year) and better signal quality.
- HTF: 1d ATR for volatility regime filter (high ATR = trending market, low ATR = range-bound).
- Volume: Current 12h volume > 1.8 * 20-period volume MA to capture institutional interest.
- Camarilla: H3 and L3 levels calculated from prior day's range.
- Entry: Long when price breaks above H3 AND 1d ATR > 20-period ATR MA AND volume spike.
         Short when price breaks below L3 AND 1d ATR > 20-period ATR MA AND volume spike.
- Exit: Price reverts to prior day's close (typical price) or loss of volume confirmation.
- Signal size: 0.25 discrete to minimize fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy combines volatility regime filtering with Camarilla pivot breakouts,
using ATR to identify trending markets where breakouts are more likely to succeed.
Volume spikes confirm institutional participation. Works in both bull and bear markets
by only taking breakout trades in high volatility regimes, avoiding false breakouts
in low volatility/choppy conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = df_1d_high[0] - df_1d_low[0]  # First bar
    tr2[0] = np.abs(df_1d_high[0] - df_1d_close[0])
    tr3[0] = np.abs(df_1d_low[0] - df_1d_close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period 1d ATR MA for regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior day's OHLC
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # Typical price (used as pivot/close reference) = (high + low + close) / 3
    h1d = df_1d['high'].values
    l1d = df_1d['low'].values
    c1d = df_1d['close'].values
    
    camarilla_h3 = c1d + 1.1 * (h1d - l1d) / 4
    camarilla_l3 = c1d - 1.1 * (h1d - l1d) / 4
    camarilla_close = (h1d + l1d + c1d) / 3  # Typical price as reference
    
    # Align HTF indicators to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_close_aligned = align_htf_to_ltf(prices, df_1d, camarilla_close)
    
    # Volatility regime filter: ATR > ATR MA (trending market)
    vol_regime = atr_1d_aligned > atr_ma_1d_aligned
    
    # Volume confirmation: current 12h volume > 1.8 * 20-period 1d volume MA
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need ATR(14)+20 MA for ATR, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike and volatility regime
            if volume_spike[i] and vol_regime[i]:
                # Bullish breakout: price > H3 AND volatility regime (trending)
                if curr_close > camarilla_h3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < L3 AND volatility regime (trending)
                elif curr_close < camarilla_l3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to typical price OR loss of volume confirmation OR volatility regime
            if (curr_close <= camarilla_close_aligned[i] or 
                not volume_spike[i] or not vol_regime[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to typical price OR loss of volume confirmation OR volatility regime
            if (curr_close >= camarilla_close_aligned[i] or 
                not volume_spike[i] or not vol_regime[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0