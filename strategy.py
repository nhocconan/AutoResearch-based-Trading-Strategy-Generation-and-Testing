#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# - Uses 1d HTF for ATR regime: ATR(14) > 1.5x ATR(50) = high volatility (breakout favorable)
# - In high volatility regime: trade Donchian(20) breakouts with momentum filter
# - In low volatility regime: avoid breakouts to prevent false signals
# - Volume confirmation: current 12h volume > 1.3x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_donchian_atr_vol_v1"
timeframe = "12h"
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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility regime: ATR(14) > 1.3 * ATR(50) = high volatility (breakout favorable)
    vol_regime_high = atr_14 > 1.3 * atr_50
    
    # Calculate 12h Donchian channels (20-period)
    # Donchian upper = highest high of last 20 periods
    # Donchian lower = lowest low of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d volatility regime to 12h timeframe (wait for completed 1d bar)
    vol_regime_high_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_high.astype(float))
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_regime_high_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Volatility regime: high volatility favors breakouts
        high_vol = vol_regime_high_aligned[i] > 0.5  # boolean as float
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: price retrace to midpoint or opposite Donchian touch
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            if close[i] < midpoint or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: price retrace to midpoint or opposite Donchian touch
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            if close[i] > midpoint or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: Donchian breakouts in high volatility regime with volume confirmation
            if volume_confirmed and high_vol:
                # Long breakout: price closes above Donchian upper
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = position_size
                # Short breakout: price closes below Donchian lower
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals