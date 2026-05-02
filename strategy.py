#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1d timeframe for signal generation with Donchian channels from 20-period
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.8x 30-period average) ensures institutional participation
# Chop regime filter from 1d timeframe avoids ranging markets (CHOP > 61.8 = range)
# Discrete position sizing (0.25) balances return and risk
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Works in bull markets via trend-aligned breakouts, in bear via chop filter avoiding false signals
# Designed for low trade frequency to minimize fee drag while capturing alpha

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    # Calculate 1d Chopiness Index (14) - trending when < 38.2, ranging when > 61.8
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
    
    # Chop = 100 * log15(sum(ATR14)/ (max(high)-min(low)) over 14 periods)
    chop = 100 * np.log15(atr14 * 14 / (max_high - min_low))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i]) or
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
                
                # Long: Price breaks above Donchian high + price > 1w EMA50 + volume confirm
                if close[i] > donch_high and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Donchian low + price < 1w EMA50 + volume confirm
                elif close[i] < donch_low and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
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