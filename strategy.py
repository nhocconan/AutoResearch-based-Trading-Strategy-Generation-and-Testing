#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for higher timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-day EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4-hour Donchian channel (20-period) for breakout signals
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (15-period on 4h)
    vol_ma15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_spike = volume > 1.8 * vol_ma15  # Require 1.8x volume for confirmation
    
    # Session filter: 08-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align higher timeframe indicators to 4-hour timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(atr14_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ma15[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr14_1d_aligned[max(0, i-49):i+1])
        vol_filter = atr14_1d_aligned[i] > 0.8 * atr_median
        
        if position == 0:
            # Long: Price breaks above Donchian high + above 1d EMA50 + volatility filter + volume spike
            if (close[i] > donch_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_filter and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below 1d EMA50 + volatility filter + volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_filter and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or trend changes
            if position == 1:
                if (close[i] < donch_low[i] or 
                    close[i] < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > donch_high[i] or 
                    close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_1dEMA50_Vol_VolFilter_Session"
timeframe = "4h"
leverage = 1.0