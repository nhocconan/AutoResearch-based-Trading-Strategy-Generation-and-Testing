#!/usr/bin/env python3
"""
6h_ElderRay_WeeklyTrend_RegimeFilter_v1
Hypothesis: Trade 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volatility regime.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
- 1w EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
- Volatility regime: ATR(14) percentile > 0.7 = high vol (trade with trend), < 0.3 = low vol (fade extremes)
- In bull trend + high vol: buy when Bull Power > 0 and rising
- In bear trend + high vol: sell when Bear Power < 0 and falling
- In low vol: fade at 2.0*ATR from EMA13 (mean reversion)
- Volume confirmation: require volume > 1.3x 20-period average
- Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
- Works in bull via trend-following, in bear via mean reversion in low vol and selective shorting.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # ATR(14) for volatility regime
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ATR percentile rank (50-period lookback) for volatility regime
    atr_percentile = np.zeros_like(atr14)
    for i in range(50, len(atr14)):
        window = atr14[i-50:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            atr_percentile[i] = (np.sum(valid <= atr14[i]) / len(valid)) * 100
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13(13), ATR(14), vol MA(20), ATR percentile(50)
    start_idx = max(13, 14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema13[i]) or
            np.isnan(atr14[i]) or
            np.isnan(atr_percentile[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend
        htf_1w_bullish = close[i] > ema50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema50_1w_aligned[i]
        
        # Determine volatility regime
        high_vol = atr_percentile[i] > 70  # ATR percentile > 70 = high volatility
        low_vol = atr_percentile[i] < 30   # ATR percentile < 30 = low volatility
        
        if position == 0:
            if htf_1w_bullish and high_vol:
                # Bull trend + high vol: trend following on bull power strength
                long_setup = (bull_power[i] > 0) and (bull_power[i] > bull_power[i-1]) and volume_spike[i]
                if long_setup:
                    signals[i] = 0.25
                    position = 1
            elif htf_1w_bearish and high_vol:
                # Bear trend + high vol: trend following on bear power weakness
                short_setup = (bear_power[i] < 0) and (bear_power[i] < bear_power[i-1]) and volume_spike[i]
                if short_setup:
                    signals[i] = -0.25
                    position = -1
            elif low_vol:
                # Low volatility: mean reversion at 2.0*ATR from EMA13
                long_setup = (close[i] < (ema13[i] - 2.0 * atr14[i])) and volume_spike[i]
                short_setup = (close[i] > (ema13[i] + 2.0 * atr14[i])) and volume_spike[i]
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if htf_1w_bullish and high_vol:
                # In bull trend + high vol: exit on bear power turning positive (weakness)
                exit_signal = bear_power[i] > 0
            elif low_vol:
                # In low vol: exit on mean reversion to EMA13 or overextension
                exit_signal = (close[i] > ema13[i]) or (close[i] > (ema13[i] + 1.5 * atr14[i]))
            else:
                # Default: exit on trend reversal or power failure
                exit_signal = (not htf_1w_bullish) or (bull_power[i] <= 0)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if htf_1w_bearish and high_vol:
                # In bear trend + high vol: exit on bull power turning negative (weakness)
                exit_signal = bull_power[i] < 0
            elif low_vol:
                # In low vol: exit on mean reversion to EMA13 or overextension
                exit_signal = (close[i] < ema13[i]) or (close[i] < (ema13[i] - 1.5 * atr14[i]))
            else:
                # Default: exit on trend reversal or power failure
                exit_signal = (not htf_1w_bearish) or (bear_power[i] >= 0)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_WeeklyTrend_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0