#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with daily volume and ATR filter
# Long when price breaks above weekly Donchian high + volume surge + ATR filter
# Short when price breaks below weekly Donchian low + volume surge + ATR filter
# Exit when price reverses back into weekly Donchian channel or volatility drops
# Designed for low frequency (~15-25 trades/year) with high conviction signals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly Donchian high/low
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any values are NaN
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_ma[i])):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter (ATR above average)
        vol_filter = atr[i] > atr_ma[i]
        
        if position == 0:  # No position - look for breakouts
            if volume_filter and vol_filter:
                # Long breakout above weekly Donchian high
                if close[i] > donchian_high_daily[i]:
                    position = 1
                    signals[i] = position_size
                # Short breakout below weekly Donchian low
                elif close[i] < donchian_low_daily[i]:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit on reversal or volatility drop
            # Exit when price returns to weekly Donchian low or volatility drops
            if close[i] < donchian_low_daily[i] or atr[i] < 0.8 * atr_ma[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit on reversal or volatility drop
            # Exit when price returns to weekly Donchian high or volatility drops
            if close[i] > donchian_high_daily[i] or atr[i] < 0.8 * atr_ma[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0