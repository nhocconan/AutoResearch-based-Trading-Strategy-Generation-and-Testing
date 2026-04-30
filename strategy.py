#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# Uses 6h primary timeframe to target 50-150 trades over 4 years (12-37/year).
# Weekly Camarilla pivots from 1d data (using previous week's OHLC) provide strong support/resistance.
# Breakouts beyond weekly R3/S3 indicate momentum moves with structure.
# Volume spike (2.0x 20-period average) confirms validity.
# Discrete sizing 0.25 balances risk and minimizes fee churn.
# Works in bull via breakout longs, in bear via breakout shorts.

name = "6h_Donchian20_Breakout_WeeklyCamarilla_R3S3_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d data for weekly Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of data
        return np.zeros(n)
    
    # Calculate weekly OHLC from daily data
    # Group by week (Monday to Sunday) using resample-like logic
    # We'll calculate weekly pivot using previous week's OHLC
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values  # Approx 5 trading days
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly Camarilla levels using previous week's OHLC
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r3 = weekly_pivot + (weekly_high - weekly_low) * 1.1 / 4.0
    weekly_s3 = weekly_pivot - (weekly_high - weekly_low) * 1.1 / 4.0
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed week)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3, additional_delay_bars=0)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3, additional_delay_bars=0)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 20)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_weekly_r3 = weekly_r3_aligned[i]
        curr_weekly_s3 = weekly_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above Donchian high AND above weekly R3
                if curr_close > curr_donchian_high and curr_close > curr_weekly_r3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below Donchian low AND below weekly S3
                elif curr_close < curr_donchian_low and curr_close < curr_weekly_s3:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Donchian low or weekly S3
            if curr_close < curr_donchian_low or curr_close < curr_weekly_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high or weekly R3
            if curr_close > curr_donchian_high or curr_close > curr_weekly_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals