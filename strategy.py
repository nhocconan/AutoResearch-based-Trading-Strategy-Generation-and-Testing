#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for trend direction (EMA50) and 1d for volume confirmation.
- Trend: 4h EMA50 slope > 0 = uptrend, < 0 = downtrend.
- Entry: In uptrend: long when price breaks above Camarilla R3 AND 1d volume > 1.5 * 20-day volume MA.
         In downtrend: short when price breaks below Camarilla S3 AND 1d volume > 1.5 * 20-day volume MA.
- Exit: Opposite Camarilla level (R3 for long exit, S3 for short exit) OR trend reversal.
- Volume confirmation: avoids low-volume false breakouts.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate EMA50 slope (trend direction) on 4h
    ema_50_slope = np.zeros_like(ema_50_aligned)
    ema_50_slope[1:] = ema_50_aligned[1:] - ema_50_aligned[:-1]
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-day volume MA on 1d
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20, additional_delay_bars=1)
    
    # 1d volume spike: current volume > 1.5 * 20-day volume MA
    volume_spike = df_1d['volume'].values > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike, additional_delay_bars=1)
    
    # Calculate Camarilla levels (using previous day's OHLC)
    # We need daily OHLC - resample 1h to daily for Camarilla calculation
    # But to avoid look-ahead, we use previous day's OHLC
    # Create daily OHLC from 1h data (using completed bars only)
    df = prices.copy()
    df['date'] = df.index.date
    daily_ohlc = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Need at least 2 days of data
    if len(daily_ohlc) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_high = daily_ohlc['high'].shift(1).values  # Previous day's high
    prev_low = daily_ohlc['low'].shift(1).values    # Previous day's low
    prev_close = daily_ohlc['close'].shift(1).values # Previous day's close
    
    # Align daily data to 1h index (using previous day's values for current day)
    # Create arrays of same length as prices, filled with previous day's Camarilla levels
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Map each 1h bar to its corresponding previous day
    dates = df.index.date
    unique_dates = daily_ohlc.index
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    
    for i in range(n):
        date = dates[i]
        if date in date_to_idx:
            idx = date_to_idx[date]
            if idx > 0:  # Have previous day data
                ph = prev_high[idx]
                pl = prev_low[idx]
                pc = prev_close[idx]
                if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
                    camarilla_r3[i] = pc + (1.1 * (ph - pl)) / 2
                    camarilla_s3[i] = pc - (1.1 * (ph - pl)) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where we have valid Camarilla levels
    start_idx = 50  # Need enough data for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_50_slope[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_slope = ema_50_slope[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                if ema_slope > 0:  # Uptrend: look for long breakout above R3
                    if curr_high > r3:
                        signals[i] = 0.20
                        position = 1
                elif ema_slope < 0:  # Downtrend: look for short breakdown below S3
                    if curr_low < s3:
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR trend reverses to downtrend
            if curr_low < s3 or ema_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 OR trend reverses to uptrend
            if curr_high > r3 or ema_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_4hEMA50Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0