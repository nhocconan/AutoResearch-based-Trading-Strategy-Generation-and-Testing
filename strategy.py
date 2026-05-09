#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Price Channel Breakout with Daily ATR Volatility Filter and Volume Spike
# Uses daily Donchian channel (20) as price channel, daily ATR(20) to measure volatility,
# and volume spike for confirmation. In high volatility regimes, breakouts are more likely
# to trend; in low volatility, they often fail. Filters out low-volatility breakouts that
# lead to whipsaws. Designed for 15-25 trades/year to minimize fee drag.
name = "6h_PriceChannelBreakout_DailyATR_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian bands and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(df_daily['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_daily['low']).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR(20) for volatility filter
    high_low = df_daily['high'] - df_daily['low']
    high_close = np.abs(df_daily['high'] - df_daily['close'].shift())
    low_close = np.abs(df_daily['low'] - df_daily['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian bands and ATR to 6h
    donchian_high_6h = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_daily, donchian_low)
    atr_6h = align_htf_to_ltf(prices, df_daily, atr)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility condition: current ATR > 1.2 x 20-period average ATR
        vol_filter = atr_6h[i] > np.nanmean(atr_6h[max(0, i-40):i]) * 1.2 if i >= 40 else False
        
        # Volume condition: current volume > 1.5 x 20-period average volume
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Break above daily Donchian high with high volatility and volume spike
            if close[i] > donchian_high_6h[i] and vol_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below daily Donchian low with high volatility and volume spike
            elif close[i] < donchian_low_6h[i] and vol_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below daily Donchian low
            if close[i] < donchian_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above daily Donchian high
            if close[i] > donchian_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals