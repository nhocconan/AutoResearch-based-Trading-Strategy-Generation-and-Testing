#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA on 6h)
# Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d close > 1d EMA34 (uptrend) AND volume > 1.5x 20-period MA
# Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND 1d close < 1d EMA34 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Opposite Elder Ray signal OR 1d trend reversal OR volume drops.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Elder Ray measures price relative to EMA for momentum; 1d EMA34 filters for higher timeframe trend; volume confirmation
# reduces false signals. Works in bull via long signals and bear via short signals when aligned with 1d trend.

name = "6h_ElderRay_1dEMA34_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    close_1d_val = close_1d[-1] if len(close_1d) > 0 else np.nan  # current 1d close (will be aligned)
    
    # Align 1d EMA34 and close to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Elder Ray components on 6h: EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine 1d trend
        is_uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Elder Ray signals
        is_bullish = bull_power[i] > 0 and bear_power[i] < 0  # High > EMA13 > Low
        is_bearish = bull_power[i] < 0 and bear_power[i] > 0  # High < EMA13 < Low
        
        # Entry logic
        if position == 0:
            # Long: Bullish Elder Ray AND 1d uptrend AND volume spike
            if is_bullish and is_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Elder Ray AND 1d downtrend AND volume spike
            elif is_bearish and is_downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Elder Ray OR 1d downtrend OR volume drops
            if is_bearish or not is_uptrend or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Elder Ray OR 1d uptrend OR volume drops
            if is_bullish or not is_downtrend or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals