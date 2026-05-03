#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) in bull trend (close > 1d EMA50) with volume spike.
# Short when Bear Power < 0 AND Bull Power > 0 in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Elder Ray measures bull/bear strength relative to EMA, effective in both trending and ranging markets.
# The 1d EMA50 filter ensures alignment with higher timeframe trend. Volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need at least 50 for EMA + 1 for current
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray Index on 6h timeframe (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        bp = bull_power[i]
        br = bear_power[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Elder Ray conditions: Bull Power > 0 AND Bear Power < 0 (market has both bull and bear energy)
        elder_long_condition = bp > 0 and br < 0
        elder_short_condition = br < 0 and bp > 0  # Same condition, but we interpret based on trend
        
        # Entry logic
        if position == 0:
            if is_bull_trend and elder_long_condition and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and elder_short_condition and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Elder Ray condition fails OR trend reversal
            if not (bp > 0 and br < 0) or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Elder Ray condition fails OR trend reversal
            if not (br < 0 and bp > 0) or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals