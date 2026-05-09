#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout (20-period) with 1w EMA trend filter and volume confirmation.
# Uses weekly Donchian channels for breakout detection, weekly EMA50 for trend direction, and volume spike for confirmation.
# Designed to capture strong trending moves while avoiding false breakouts in choppy markets.
# Target: 7-25 trades/year (30-100 total over 4 years) on 1d timeframe.
name = "1d_WeeklyDonchian20_1wEMA50_VolumeConfirm"
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
    
    # Get weekly data for EMA50 trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = np.full(len(df_1w), np.nan)
    donchian_low = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate weekly volume average (20-period) for spike detection
    volume_1w = df_1w['volume'].values
    vol_ma = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        vol_ma[i] = np.mean(volume_1w[i-20:i])
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema = ema_50_1w_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol = volume[i]
        vol_avg = vol_ma_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high AND price > weekly EMA50 (uptrend) AND volume > 1.5x average
            if price > upper and price > ema and vol > 1.5 * vol_avg:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low AND price < weekly EMA50 (downtrend) AND volume > 1.5x average
            elif price < lower and price < ema and vol > 1.5 * vol_avg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly Donchian low OR trend reverses (price < weekly EMA50)
            if price < lower or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly Donchian high OR trend reverses (price > weekly EMA50)
            if price > upper or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals