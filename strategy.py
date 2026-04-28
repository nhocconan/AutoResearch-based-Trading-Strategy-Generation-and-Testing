#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d confluence filter. Uses 4h Donchian breakout for direction,
# 1d EMA50 for trend filter, and volume confirmation. 1h only for entry timing precision.
# Targets 15-30 trades/year (60-120 over 4 years) by requiring 4h+1d alignment + volume spike.
# Discrete position size 0.20 to limit drawdown. Works in bull/bear via multi-TF trend filter.

name = "1h_DonchianBreakout_4hDirection_1dEMA50_Trend_Volume_v1"
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
    
    # Get HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_ma_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_ma_20_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_ma_20_4h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_4h_aligned[i]) or 
            np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters: 4h price vs Donchian, 1d price vs EMA50
        price_above_4h_donchian = close[i] > donchian_high_4h_aligned[i]
        price_below_4h_donchian = close[i] < donchian_low_4h_aligned[i]
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = price_above_4h_donchian and volume_spike[i]
        short_breakout = price_below_4h_donchian and volume_spike[i]
        
        # Require alignment: 4h breakout direction must match 1d trend
        long_signal = long_breakout and price_above_1d_ema
        short_signal = short_breakout and price_below_1d_ema
        
        # Exit conditions: opposite 4h Donchian level or 1d trend reversal
        long_exit = close[i] < donchian_low_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]
        short_exit = close[i] > donchian_high_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals