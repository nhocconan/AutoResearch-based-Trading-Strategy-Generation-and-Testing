#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation
# - Uses 1d HTF for ATR-based volatility regime: ATR(14) > 20-period median = high volatility
# - In high volatility: trade Donchian(20) breakouts with momentum filter (close > open)
# - In low volatility: avoid breakouts to prevent false signals in choppy markets
# - Volume confirmation: current 4h volume > 1.3x 20-period average to filter weak breakouts
# - Fixed position size 0.25 to control drawdown and minimize fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Designed to work in both bull (breakouts continue) and bear (breakouts reverse) markets
#   by using volatility filter to avoid false breakouts in ranging/low-vol periods

name = "4h_1d_donchian_vol_filter_v1"
timeframe = "4h"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 20-period median of 1d ATR for volatility regime threshold
    atr_median_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    
    # Align 1d ATR and median to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_median_20_aligned = align_htf_to_ltf(prices, df_1d, atr_median_20)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_median_20_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volatility regime: 1d ATR > 20-period median = high volatility (good for breakouts)
        high_volatility = atr_1d_aligned[i] > atr_median_20_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Momentum filter: close > open for long, close < open for short (avoid weak breakouts)
        bullish_momentum = close[i] > prices['open'].iloc[i]
        bearish_momentum = close[i] < prices['open'].iloc[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low or volatility drops
            if close[i] < donchian_low[i] or not high_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high or volatility drops
            if close[i] > donchian_high[i] or not high_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: only in high volatility with volume confirmation
            if high_volatility and volume_confirmed:
                # Long breakout: price above Donchian high with bullish momentum
                if close[i] > donchian_high[i] and bullish_momentum:
                    position = 1
                    signals[i] = position_size
                # Short breakout: price below Donchian low with bearish momentum
                elif close[i] < donchian_low[i] and bearish_momentum:
                    position = -1
                    signals[i] = -position_size
    
    return signals