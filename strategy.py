#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Trend filter (EMA50) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h: Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Upper and lower bands
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema50_1d_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + uptrend + volume confirmation
            if (close_val > donchian_high_val and    # Breakout above upper band
                close_val > ema_val and              # Price above 1d EMA50 (uptrend)
                vol_ratio_val > 1.5):                # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + downtrend + volume confirmation
            elif (close_val < donchian_low_val and   # Breakdown below lower band
                  close_val < ema_val and            # Price below 1d EMA50 (downtrend)
                  vol_ratio_val > 1.5):              # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or trend reversal
            if (close_val < donchian_low_val or    # Breakdown below lower band
                close_val < ema_val):              # Price below 1d EMA50 (trend reversal)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or trend reversal
            if (close_val > donchian_high_val or   # Breakout above upper band
                close_val > ema_val):              # Price above 1d EMA50 (trend reversal)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals