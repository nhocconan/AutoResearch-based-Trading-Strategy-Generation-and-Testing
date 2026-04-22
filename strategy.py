#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly trend filter + daily price channel breakout with volume confirmation
# Uses weekly EMA50 for trend direction and daily Donchian breakout for entries
# Works in bull (trend following) and bear (counter-trend when price deviates from weekly trend)
# Target: 20-40 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA50 for trend filter (no look-ahead)
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Load daily data for price channels
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily Donchian channels (20-period) - using prior day's data only
    # Roll by 1 to avoid look-ahead (use previous day's high/low for today's breakout)
    prev_high_daily = np.roll(high_daily, 1)
    prev_low_daily = np.roll(low_daily, 1)
    prev_high_daily[0] = np.nan
    prev_low_daily[0] = np.nan
    
    # Calculate 20-period rolling max/min of previous day's data
    donchian_high = pd.Series(prev_high_daily).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low_daily).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    
    # Daily volume spike filter
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any data is not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema50 = ema50_aligned[i]
        dch_high = donchian_high_aligned[i]
        dch_low = donchian_low_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily Donchian high + volume + price above weekly EMA50
            if price > dch_high and vol > 1.5 * vol_ma and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low + volume + price below weekly EMA50
            elif price < dch_low and vol > 1.5 * vol_ma and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through the opposite Donchian level
            if position == 1 and price < dch_low:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > dch_high:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_WeeklyEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0