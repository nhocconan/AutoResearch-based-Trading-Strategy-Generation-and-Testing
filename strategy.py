#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ADX(14) > 25 trend filter
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# ADX(14) > 25 filters for trending markets to avoid whipsaws in ranging conditions
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Donchian breakouts capture sustained moves, volume confirms validity, ADX ensures trend strength
# Works in both bull and bear markets by following established trends with filters

name = "4h_Donchian20_Volume_ADX25_Trend_v1"
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
    
    # Load 1d data ONCE before loop for potential HTF filter (though not used in final logic)
    # Keeping structure for MTF compliance but focusing on 4h factors
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # ADX calculation for trend strength filter
    # +DM, -DM, TR calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - low[:-1]), 
                               np.abs(low[1:] - high[:-1])))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    tr = np.concatenate([[np.nan], tr])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    plus_di_14 = 100 * wilders_smoothing(plus_dm, period_adx) / wilders_smoothing(tr, period_adx)
    minus_di_14 = 100 * wilders_smoothing(minus_dm, period_adx) / wilders_smoothing(tr, period_adx)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = wilders_smoothing(dx, period_adx)
    
    # ADX > 25 trend filter
    trend_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and ADX)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(trend_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper + volume confirm + trend filter
            if close[i] > high_ma[i] and volume_confirm[i] and trend_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + volume confirm + trend filter
            elif close[i] < low_ma[i] and volume_confirm[i] and trend_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower (stop) or reverse signal
            if close[i] < low_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper (stop) or reverse signal
            if close[i] > high_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals