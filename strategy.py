#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above Donchian(20) high AND ATR(14) < ATR(50) (low volatility regime) AND volume > 1.5x average volume
# Short when price breaks below Donchian(20) low AND ATR(14) < ATR(50) AND volume > 1.5x average volume
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 20-40 trades/year.
# Donchian channels provide clear breakout levels; ATR regime filter avoids whipsaws in high volatility; volume confirmation ensures participation.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following price structure with volatility filter.

name = "4h_DonchianBreakout_ATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50) for regime filter
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR regime: low volatility when ATR(14) < ATR(50)
    atr_regime = atr_14 < atr_50
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Calculate Donchian channels on 4h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 30)  # warmup for Donchian and volume
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_spike = volume_spike[i]
        curr_atr_regime = atr_regime_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND low volatility regime AND volume spike
            if (curr_close > donchian_high[i] and 
                curr_atr_regime and 
                curr_vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND low volatility regime AND volume spike
            elif (curr_close < donchian_low[i] and 
                  curr_atr_regime and 
                  curr_vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low (opposite side) OR high volatility regime
            if (curr_close < donchian_low[i] or not curr_atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (opposite side) OR high volatility regime
            if (curr_close > donchian_high[i] or not curr_atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals