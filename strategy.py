#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation (2.0x 20-period average), and chop regime filter (CHOP < 61.8)
# Uses 4h timeframe for signal generation with Donchian channels from 20-period
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation ensures institutional participation
# Chop regime filter avoids ranging markets (CHOP > 61.8 = range)
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via chop filter avoiding false signals and trend filter preventing shorts in strong uptrends

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Calculate 4h Chopiness Index (14) - trending when < 38.2, ranging when > 61.8
    # True Range
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR14
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Max high and min low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14)/ (max(high)-min(low)) over 14 periods) / log10(14)
    # Using log10 for stability: CHOP = 100 * log10(atr14 * 14 / (max_high - min_low)) / log10(14)
    chop = 100 * np.log10(atr14 * 14 / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when Chop < 61.8 (not strongly ranging)
        if chop[i] > 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Calculate Donchian channels for 20-period (need 20 bars of history)
            if i >= 20:
                # Donchian high: highest high over past 20 periods (excluding current)
                donch_high = np.max(high[i-20:i])
                # Donchian low: lowest low over past 20 periods (excluding current)
                donch_low = np.min(low[i-20:i])
                
                # Long: Price breaks above Donchian high + price > 1d EMA34 + volume confirm
                if close[i] > donch_high and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Donchian low + price < 1d EMA34 + volume confirm
                elif close[i] < donch_low and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low (20-period) or reverse signal
            if i >= 20:
                donch_low = np.min(low[i-20:i])
                if close[i] < donch_low:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high (20-period) or reverse signal
            if i >= 20:
                donch_high = np.max(high[i-20:i])
                if close[i] > donch_high:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals