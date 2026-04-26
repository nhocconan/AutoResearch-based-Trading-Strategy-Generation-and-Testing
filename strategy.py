#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm
Hypothesis: 6h Donchian(20) breakouts aligned with weekly Camarilla pivot direction (R4/S4 for trend, R3/S3 for mean reversion) and volume confirmation (>1.5x average) captures institutional moves while filtering noise. Works in bull/bear via weekly structure alignment. Designed for 6h to target 12-37 trades/year with discrete sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    camarilla_r4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_s4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align weekly levels to 6h (wait for completed weekly bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Weekly trend: price vs weekly EMA50
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation (24-period SMA = 1d * 4 = 4d)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of Donchian(20), weekly EMA(50), volume(24)
    start_idx = max(20, 50, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        high_val = high[i]
        low_val = low[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        upper_donchian = highest_high[i]
        lower_donchian = lowest_low[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r3_val) or 
            np.isnan(s3_val) or np.isnan(r4_val) or np.isnan(s4_val) or
            np.isnan(upper_donchian) or np.isnan(lower_donchian)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Determine weekly regime
        price_above_weekly_ema = close_val > ema_val
        price_below_weekly_ema = close_val < ema_val
        
        # Long conditions
        # Breakout long: price breaks above weekly R4 with weekly uptrend and volume
        breakout_long = (high_val > r4_val) and price_above_weekly_ema and volume_confirmed
        # Mean reversion long: price pulls back to weekly S3 with weekly uptrend and volume
        mean_reversion_long = (low_val <= s3_val) and price_above_weekly_ema and volume_confirmed
        
        # Short conditions
        # Breakout short: price breaks below weekly S4 with weekly downtrend and volume
        breakout_short = (low_val < s4_val) and price_below_weekly_ema and volume_confirmed
        # Mean reversion short: price pulls back to weekly R3 with weekly downtrend and volume
        mean_reversion_short = (high_val >= r3_val) and price_below_weekly_ema and volume_confirmed
        
        # Exit conditions: price returns to weekly EMA50
        long_exit = position == 1 and close_val <= ema_val
        short_exit = position == -1 and close_val >= ema_val
        
        if (breakout_long or mean_reversion_long) and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif (breakout_short or mean_reversion_short) and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm"
timeframe = "6h"
leverage = 1.0