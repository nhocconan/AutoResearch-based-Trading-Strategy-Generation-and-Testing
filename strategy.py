#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
# In weekly uptrend: long on Donchian upper break; in weekly downtrend: short on Donchian lower break.
# Uses volume > 1.5x 20-period average for confirmation. Avoids counter-trend trades.
# Target: 15-25 trades/year by requiring weekly trend alignment + breakout + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend
    weekly_close = df_1w['close'].values
    ema_34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Pre-compute daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(weekly_trend[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Weekly trend: above EMA = uptrend, below = downtrend
        is_uptrend = price > weekly_trend[i]
        is_downtrend = price < weekly_trend[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol > 1.5 * vol_ma[i]
        
        if position == 0:
            if is_uptrend and volume_confirm:
                # Weekly uptrend: long on Donchian upper break
                if price > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and volume_confirm:
                # Weekly downtrend: short on Donchian lower break
                if price < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit long on breakdown below Donchian low
                if price < donchian_low[i]:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit short on breakout above Donchian high
                if price > donchian_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0