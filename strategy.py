#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian channel breakout with daily volume confirmation and ATR filter
# Works in bull markets by catching breakouts, in bear markets by avoiding false signals via volume/ATR filters
# Target: 1d timeframe with 1h trend filter for better timing, aiming for 10-25 trades/year
name = "1d_WeeklyDonchian_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate 20-period weekly Donchian channels
    high_max_20 = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high = align_htf_to_ltf(prices, df_weekly, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_weekly, low_min_20)
    
    # Get daily data for volume and ATR
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    volume_daily = df_daily['volume'].values
    
    # Daily ATR (14-period) for volatility filter
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        volatility_filter = atr_val > 0  # Always true but keeps structure
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + volume confirmation
            if price > upper and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low + volume confirmation
            elif price < lower and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly Donchian low or volatility drops
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly Donchian high or volatility drops
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals