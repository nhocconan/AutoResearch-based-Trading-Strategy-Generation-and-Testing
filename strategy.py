#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian Breakout with 4h Trend Filter and Volume Spike
# - Uses 4h Donchian channels (upper/lower bands) for trend direction
# - Breakout above 1h upper band with 4h uptrend or below lower band with 4h downtrend
# - Volume spike confirms breakout strength
# - Session filter (08-20 UTC) to reduce noise trades
# - Position size 0.20 to manage drawdown in bear markets
# - Target: 15-35 trades/year to minimize fee drag on 1h timeframe

name = "1h_DonchianBreakout_4hTrend_Volume_Session"
timeframe = "1h"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for Donchian channels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    n4h = len(close_4h)
    donchian_upper_4h = np.full(n4h, np.nan)
    donchian_lower_4h = np.full(n4h, np.nan)
    
    for i in range(20, n4h):
        high_window = high_4h[i-20:i]
        low_window = low_4h[i-20:i]
        donchian_upper_4h[i] = np.max(high_window)
        donchian_lower_4h[i] = np.min(low_window)
    
    # Align Donchian channels to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h upper band with 4h uptrend + volume spike
            long_cond = (close[i] > donchian_upper_4h_aligned[i] and 
                        ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below 4h lower band with 4h downtrend + volume spike
            short_cond = (close[i] < donchian_lower_4h_aligned[i] and 
                         ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h lower band (trend reversal)
            if close[i] < donchian_lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 4h upper band (trend reversal)
            if close[i] > donchian_upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals