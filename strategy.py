#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX trend regime.
- Williams %R(14): Long when < -80 (oversold), Short when > -20 (overbought) in 6h.
- Trend filter: Only trade counter to 1d ADX regime - mean revert when ADX < 25 (range), 
  trend follow when ADX > 25 (trend) but only if Williams %R confirms pullback in trend direction.
- Volume confirmation: current volume > 1.5x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying oversold dips in uptrend (ADX>25) or mean reversion in range (ADX<25).
- Works in bear via selling overbought rallies in downtrend (ADX>25) or mean reversion in range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R on 6h (primary timeframe)
    def williams_r(high, low, close, period):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d
    def calculate_adx(high, low, close, period):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Williams %R needs 14, plus buffers
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based logic
            if adx_1d_aligned[i] < 25:  # Range regime - mean revert
                # Long when oversold, short when overbought
                if wr[i] < -80 and volume_spike[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif wr[i] > -20 and volume_spike[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
            else:  # Trend regime - trend follow on pullbacks
                # Need to determine trend direction from price action
                if i >= 10:
                    # Simple trend: compare current close to 10-period ago
                    trend_up = close[i] > close[i-10]
                    if trend_up:  # Uptrend - look for pullbacks to go long
                        if wr[i] < -50 and volume_spike[i]:  # Pullback into oversold territory
                            signals[i] = 0.25
                            position = 1
                    else:  # Downtrend - look for bounces to go short
                        if wr[i] > -50 and volume_spike[i]:  # Pullback into overbought territory
                            signals[i] = -0.25
                            position = -1
        elif position == 1:
            # Long exit: Williams %R returns to neutral or overbought
            if wr[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to neutral or oversold
            if wr[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_EXTREME_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0