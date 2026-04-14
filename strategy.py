#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian Breakout with Daily Volume Confirmation
# 1. Entry: Price breaks weekly Donchian(20) channel with daily volume > 1.5x 20-day average
# 2. Exit: Price returns to weekly Donchian midline or volume drops below average
# 3. Position size: 0.25 for breakouts, 0 for exits
# Rationale: Weekly channels capture major trends, volume confirms institutional interest,
# works in bull/bear by catching breakouts in either direction. Target: 15-25 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly upper/lower bands
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_daily = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    
    # Daily volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    # Start after enough data for weekly Donchian and volume MA
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or
            np.isnan(donchian_mid_daily[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long breakout: price above weekly upper band with volume confirmation
            if price > donchian_high_daily[i] and vol > 1.5 * vol_ma[i]:
                position = 1
                signals[i] = position_size
            # Short breakout: price below weekly lower band with volume confirmation
            elif price < donchian_low_daily[i] and vol > 1.5 * vol_ma[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly midline
            if price <= donchian_mid_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly midline
            if price >= donchian_mid_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "Weekly_Donchian_Breakout_DailyVolume"
timeframe = "1d"
leverage = 1.0