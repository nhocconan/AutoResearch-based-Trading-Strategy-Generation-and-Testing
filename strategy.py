#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with daily volume confirmation and weekly volatility regime
# The Williams Alligator (13/8/5 SMAs with 8/5/3 offsets) identifies trends when jaws/lips/teeth align
# Long when teeth > jaws > lips with volume > 1.5x 20-day average and weekly volatility < 30th percentile
# Short when teeth < jaws < lips with volume > 1.5x 20-day average and weekly volatility < 30th percentile
# Exit when Alligator lines cross or volatility exceeds 70th percentile
# Targets 20-40 trades/year to minimize fee decay while capturing sustained trends

name = "12h_Williams_Alligator_Vol_VolRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Get weekly data for volatility regime filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h close
    # Jaw: 13-period SMA, shifted by 8 bars
    jaw = np.full_like(close, np.nan)
    for i in range(12, len(close)):
        jaw[i] = np.mean(close[i-12:i+1])
    jaw = np.roll(jaw, 8)
    
    # Teeth: 8-period SMA, shifted by 5 bars
    teeth = np.full_like(close, np.nan)
    for i in range(7, len(close)):
        teeth[i] = np.mean(close[i-7:i+1])
    teeth = np.roll(teeth, 5)
    
    # Lips: 5-period SMA, shifted by 3 bars
    lips = np.full_like(close, np.nan)
    for i in range(4, len(close)):
        lips[i] = np.mean(close[i-4:i+1])
    lips = np.roll(lips, 3)
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_20 = np.full_like(daily_volume, np.nan)
    for i in range(len(daily_volume)):
        if i < 20:
            vol_ma_20[i] = np.mean(daily_volume[max(0, i-19):i+1]) if i >= 0 else daily_volume[i]
        else:
            vol_ma_20[i] = np.mean(daily_volume[i-19:i+1])
    
    # Calculate weekly volatility percentile (using ATR-based volatility)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR14
    atr14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            atr14[i] = np.mean(tr[max(0, i-13):i+1]) if i >= 0 else tr[i]
        else:
            atr14[i] = np.mean(tr[i-13:i+1])
    
    # Volatility percentile rank (using 50-period lookback)
    vol_rank = np.full_like(atr14, np.nan)
    for i in range(50, len(atr14)):
        window = atr14[i-50:i+1]
        if len(window) > 0 and not np.all(np.isnan(window)):
            current = atr14[i]
            if not np.isnan(current):
                # Calculate percentile rank
                rank = np.sum(~np.isnan(window) & (window <= current)) / np.sum(~np.isnan(window)) * 100
                vol_rank[i] = rank
    
    # Align indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    vol_rank_aligned = align_htf_to_ltf(prices, df_weekly, vol_rank)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for Alligator and vol rank
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_rank_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find the most recent completed daily bar for volume filter
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        
        if idx_daily < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day average
        vol_daily_current = df_daily.iloc[idx_daily]['volume']
        vol_filter = vol_daily_current > 1.5 * vol_ma_20_aligned[i]
        
        # Volatility regime: < 30th percentile = low volatility (trending)
        vol_regime = vol_rank_aligned[i] < 30
        
        if position == 0:
            # Look for Alligator alignment with volume confirmation and low volatility regime
            # Long: teeth > jaws > lips (bullish alignment)
            if teeth_aligned[i] > jaw_aligned[i] and jaw_aligned[i] > lips_aligned[i]:
                if vol_filter and vol_regime:
                    signals[i] = 0.25
                    position = 1
            # Short: teeth < jaws < lips (bearish alignment)
            elif teeth_aligned[i] < jaw_aligned[i] and jaw_aligned[i] < lips_aligned[i]:
                if vol_filter and vol_regime:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Alligator lines cross or volatility exceeds 70th percentile
            if (teeth_aligned[i] <= jaw_aligned[i] or jaw_aligned[i] <= lips_aligned[i] or
                vol_rank_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross or volatility exceeds 70th percentile
            if (teeth_aligned[i] >= jaw_aligned[i] or jaw_aligned[i] >= lips_aligned[i] or
                vol_rank_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals