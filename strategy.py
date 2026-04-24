#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d volatility regime filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR-based volatility regime.
- High volatility (ATR(20) > 1.5 * ATR(50)) favors breakout strategy; low volatility favors mean reversion at Camarilla H3/L3 levels.
- Entry: Long when price breaks above Camarilla R3 AND high volatility regime (bullish breakout in high vol).
         Short when price breaks below Camarilla S3 AND high volatility regime (bearish breakout in high vol).
         In low volatility regime: Long when price touches Camarilla S3 AND reverses up (close > low).
                                 Short when price touches Camarilla R3 AND reverses down (close < high).
- Exit: Opposite Camarilla breakout (R3/S3) or volatility regime shift.
- Volume confirmation: current volume > 2.0 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for ATR and Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR (20-period and 50-period) on 1d for volatility regime
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: high vol when ATR(20) > 1.5 * ATR(50)
    vol_regime_high = atr_20 > (1.5 * atr_50)
    
    # Calculate Camarilla pivot levels (R3, S3) on 1d
    # Camarilla: based on previous day's OHLC
    prev_close = pd.Series(df_1d['close']).shift(1)
    prev_high = pd.Series(df_1d['high']).shift(1)
    prev_low = pd.Series(df_1d['low']).shift(1)
    
    # Camarilla R3 and S3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align 1d indicators to 4h
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_high.astype(float))
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1d bars for ATR(50) and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_regime_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_high = vol_regime_aligned[i] > 0.5  # True if high volatility regime
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if vol_high:  # High volatility regime: breakout strategy
                    # Bullish breakout: price closes above Camarilla R3
                    if curr_close > camarilla_r3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below Camarilla S3
                    elif curr_close < camarilla_s3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Low volatility regime: mean reversion at extremes
                    # Long when price touches Camarilla S3 and shows reversal (close > low)
                    if curr_low <= camarilla_s3_aligned[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches Camarilla R3 and shows reversal (close < high)
                    elif curr_high >= camarilla_r3_aligned[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S3 OR volatility regime shifts to low vol
            if curr_close < camarilla_s3_aligned[i] or not vol_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla R3 OR volatility regime shifts to low vol
            if curr_close > camarilla_r3_aligned[i] or not vol_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dVolRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0