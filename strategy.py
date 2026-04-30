#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Donchian breakouts capture strong momentum moves; 12h EMA50 ensures alignment with medium-term trend
# Volume spike (2.0x 96-period average) confirms institutional participation
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull markets via upside breakouts and bear markets via downside breakouts with trend filter.

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 96-period average (96*4h = 384h = 16 days)
    vol_ma_96 = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    volume_spike = volume > (2.0 * vol_ma_96)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 96)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_96[i]):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Calculate Donchian channels (20-period)
        lookback = 20
        if i < lookback:
            signals[i] = 0.0
            continue
        highest_high = np.max(high[i-lookback:i])
        lowest_low = np.min(low[i-lookback:i])
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if curr_volume_spike:
                # Bullish entry: break above Donchian high with price > 12h EMA50
                if curr_close > highest_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below Donchian low with price < 12h EMA50
                elif curr_close < lowest_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below Donchian low OR trend reverses (price < 12h EMA50)
            if curr_close < lowest_low or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high OR trend reverses (price > 12h EMA50)
            if curr_close > highest_high or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals