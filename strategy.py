#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and ATR Filter
Hypothesis: On daily charts, weekly Donchian channel breakouts combined with volume spikes
and ATR-based volatility filtering capture strong momentum moves in BTC/ETH while
avoiding false breakouts. The weekly timeframe provides a strong trend filter that works
in both bull and bear markets by identifying significant breakouts from longer-term ranges.
Target: 10-20 trades/year to minimize fee drag and ensure robustness across market regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (20-week lookback)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    donchian_high = weekly_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe (wait for weekly close)
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 2.0x 20-day average (high threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for weekly Donchian and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_daily[i]
        lower = donchian_low_daily[i]
        atr_val = atr[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above weekly Donchian high with volume and sufficient volatility
            if price > upper and vol_ok and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with volume and sufficient volatility
            elif price < lower and vol_ok and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns to weekly Donchian middle or volatility drops significantly
            mid = (upper + lower) / 2
            if price < mid or atr_val < 0.3 * atr[i-1]:  # Strong volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to weekly Donchian middle or volatility drops significantly
            mid = (upper + lower) / 2
            if price > mid or atr_val < 0.3 * atr[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0