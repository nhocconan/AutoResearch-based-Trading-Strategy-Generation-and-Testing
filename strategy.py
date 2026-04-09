#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# - Uses 1d EMA13 for trend direction (price > EMA13 = uptrend, < = downtrend)
# - 6h Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13
# - Long when Bull Power > 0 and rising (2-bar momentum) in uptrend
# - Short when Bear Power < 0 and falling (2-bar momentum) in downtrend
# - Volume confirmation: current 6h volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)

name = "6h_1d_elder_ray_trend_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for trend filter
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1d EMA13 to 6h timeframe (wait for completed 1d bar)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Pre-compute 6h EMA13 for Elder Ray calculation
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13_6h  # Bull Power = High - EMA13
    bear_power = low - ema_13_6h   # Bear Power = Low - EMA13
    
    # Calculate 2-bar momentum for Elder Ray
    bull_power_momentum = bull_power - np.roll(bull_power, 2)
    bear_power_momentum = bear_power - np.roll(bear_power, 2)
    # Set first 2 values to NaN (not enough history)
    bull_power_momentum[:2] = np.nan
    bear_power_momentum[:2] = np.nan
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_13_6h[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_momentum[i]) or np.isnan(bear_power_momentum[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price > 1d EMA13 = uptrend, < = downtrend
        uptrend = close[i] > ema_13_1d_aligned[i]
        downtrend = close[i] < ema_13_1d_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Bull Power becomes negative (momentum loss)
            if bull_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Bear Power becomes positive (momentum loss)
            if bear_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry conditions with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: Bull Power > 0 AND rising momentum in uptrend
                if uptrend and bull_power[i] > 0 and bull_power_momentum[i] > 0:
                    position = 1
                    signals[i] = position_size
                # Short: Bear Power < 0 AND falling momentum in downtrend
                elif downtrend and bear_power[i] < 0 and bear_power_momentum[i] < 0:
                    position = -1
                    signals[i] = -position_size
    
    return signals