#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Daily Trend and Volume Confirmation
# Hypothesis: Donchian(20) breakouts capture strong momentum moves.
# Daily trend filter (EMA50) ensures alignment with higher-timeframe momentum.
# Volume confirmation (>1.5x average) filters weak breakouts.
# Designed for 12h timeframe with low trade frequency (12-37/year).
# Works in bull via long breakouts + daily uptrend + volume,
# in bear via short breakouts + daily downtrend + volume.

name = "12h_donchian20_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    def donchian_channels(high, low, lookback=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(lookback - 1, len(high)):
            upper[i] = high[i-lookback+1:i+1].max()
            lower[i] = low[i-lookback+1:i+1].min()
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # Daily trend filter: EMA(50) of daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_confirm[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower OR daily trend turns bearish
            if close[i] < lower[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper OR daily trend turns bullish
            if close[i] > upper[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian upper with daily uptrend
                if close[i] > upper[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower with daily downtrend
                elif close[i] < lower[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals