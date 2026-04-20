#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly trend filter and volume confirmation
# Works in bull/bear by capturing breakouts in trending markets and avoiding false signals
# Low trade frequency (target: 10-25/year) to minimize fee drag
name = "1d_Donchian20_WeeklyTrend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    close_weekly = df_weekly['close'].values
    ema_34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Daily Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    
    # Upper band: highest high of last 20 days
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily average volume (20-period) for volume confirmation
    vol = prices['volume'].values
    vol_avg = pd.Series(vol).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # Need enough data for weekly EMA34 and daily indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        weekly_trend = ema_34_weekly_aligned[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        volume_avg = vol_avg[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if np.isnan(weekly_trend) or np.isnan(upper_band) or np.isnan(lower_band) or np.isnan(volume_avg):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.3x daily average volume
        vol_spike = current_volume > 1.3 * volume_avg
        
        if position == 0:
            # Long: price breaks above upper band with weekly uptrend and volume spike
            if current_close > upper_band and current_close > weekly_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below lower band with weekly downtrend and volume spike
            elif current_close < lower_band and current_close < weekly_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below lower band or trend reversal
            if current_close < lower_band or current_close < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band or trend reversal
            if current_close > upper_band or current_close > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals