#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian channel breakout with daily ATR volatility filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets, while ATR filter avoids false signals in low volatility.
# Weekly timeframe provides stable structure for long-term trends. Volume confirmation ensures institutional participation.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following the trend.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for Donchian channels ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(high, np.nan, dtype=float)
        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
        return upper, lower
    
    donchian_upper_1w, donchian_lower_1w = calculate_donchian(high_1w, low_1w, 20)
    donchian_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    
    # === Daily data for ATR and volume filters ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # ATR (14-period)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        if len(high) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 20-day average volume on daily data
    volume_1d_series = pd.Series(volume_1d)
    vol_avg20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    position = 0
    warmup = max(20, 20)  # For Donchian and ATR
    
    for i in range(warmup, n):
        if (np.isnan(donchian_upper_1w_aligned[i]) or 
            np.isnan(donchian_lower_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper + volatility expansion + volume confirmation
            if close[i] > donchian_upper_1w_aligned[i] and atr_1d_aligned[i] > 1.2 * atr_1d_aligned[i-1] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower + volatility expansion + volume confirmation
            elif close[i] < donchian_lower_1w_aligned[i] and atr_1d_aligned[i] > 1.2 * atr_1d_aligned[i-1] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below weekly Donchian lower
            if close[i] < donchian_lower_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above weekly Donchian upper
            if close[i] > donchian_upper_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_ATR_Volume"
timeframe = "1d"
leverage = 1.0