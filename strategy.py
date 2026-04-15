#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 100:
        return np.zeros(n)
    
    # Weekly close array
    close_weekly = df_weekly['close'].values
    
    # Weekly ATR(14) for volatility filter
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    tr1 = high_weekly[1:] - low_weekly[1:]
    tr2 = np.abs(high_weekly[1:] - close_weekly[:-1])
    tr3 = np.abs(low_weekly[1:] - close_weekly[:-1])
    tr_weekly = np.concatenate([[np.max([high_weekly[0] - low_weekly[0], 
                                          np.abs(high_weekly[0] - close_weekly[0]),
                                          np.abs(low_weekly[0] - close_weekly[0])])], 
                               np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_weekly = pd.Series(tr_weekly).rolling(window=14, min_periods=14).mean().values
    
    # Weekly volatility regime: ATR(14) > ATR(50) = high volatility (trending)
    atr_50_weekly = pd.Series(tr_weekly).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr_weekly > atr_50_weekly  # True when trending
    
    # Weekly Donchian channels (20-period)
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    dc_upper_weekly, dc_lower_weekly = donchian_channels(high_weekly, low_weekly, 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Get aligned weekly indicators
        dc_upper_aligned = align_htf_to_ltf(prices, df_weekly, dc_upper_weekly)[i]
        dc_lower_aligned = align_htf_to_ltf(prices, df_weekly, dc_lower_weekly)[i]
        vol_regime_aligned = align_htf_to_ltf(prices, df_weekly, vol_regime.astype(float))[i]
        
        # Check for NaN values
        if (np.isnan(dc_upper_aligned) or np.isnan(dc_lower_aligned) or 
            np.isnan(vol_regime_aligned)):
            continue
        
        # Volatility filter: only trade in trending regimes
        if vol_regime_aligned < 0.5:  # Not in trending regime
            continue
        
        if position == 0:  # No position - look for breakout entries
            # Long: price breaks above weekly Donchian upper
            if close[i] > dc_upper_aligned:
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly Donchian lower
            elif close[i] < dc_lower_aligned:
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below lower band
            if close[i] < dc_lower_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above upper band
            if close[i] > dc_upper_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyDonchianBreakout_VolRegime"
timeframe = "12h"
leverage = 1.0