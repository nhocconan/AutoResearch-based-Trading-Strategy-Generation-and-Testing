#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h primary with 4h/1d HTF - 4h Donchian breakout + 1d volume confirmation
    # Uses higher timeframe for signal direction (4h trend + 1d regime) and 1h for precise entry timing
    # Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF Donchian channels (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for HTF volume confirmation and regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period) for trend
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filter
    def calculate_atr(high, low, close, window=14):
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        return pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, window=14)
    atr_ma_10 = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align HTF indicators to 1h primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    atr_ma_10_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size (discrete level to minimize fee churn)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(atr_ma_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC to reduce noise
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation: current 1h volume > 1.2x 20-period 1d average
        # Note: comparing 1h volume to 1d average volume (scaled appropriately)
        volume_confirmed = volume[i] > 1.2 * (vol_avg_20_aligned[i] / 24)  # 1d avg / 24 ≈ 1h avg
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = True  # Simplified - using price action instead of ATR for 1h
        
        # Breakout conditions using 4h Donchian on 1h close
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and vol_filter and in_session
        enter_short = breakout_down and volume_confirmed and vol_filter and in_session
        
        # Exit conditions: price returns to 4h Donchian middle (mean reversion within trend)
        donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        exit_long = position == 1 and close[i] <= donchian_mid
        exit_short = position == -1 and close[i] >= donchian_mid
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_donchian_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0