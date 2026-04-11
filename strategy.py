#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d/1w trend filter + volume confirmation
# - Donchian(20) breakout on 12h: long when price > 20-period high, short when price < 20-period low
# - Trend filter: 1d close > 1w EMA50 for long bias, 1d close < 1w EMA50 for short bias
# - Volume confirmation: 12h volume > 1.5x 20-period volume SMA
# - Exit when price crosses Donchian midpoint or volume drops below average
# - Target: 15-30 trades/year to minimize fee drag while capturing strong trends

name = "12h_1d_1w_donchian_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for Donchian and volume (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return signals
    
    # Load 1d data ONCE before loop for trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for EMA50 trend (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian to 12h timeframe (already in 12h, but align for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Pre-compute 12h volume SMA for confirmation
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 1d close for trend filter
    close_1d = df_1d['close'].values
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_sma_20_12h_aligned[i]) or
            np.isnan(close_1d[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        volume_12h_current = df_12h['volume'].values
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_current)
        vol_confirm = volume_12h_aligned[i] > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Price levels
        price_12h = close_12h[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        dm = donchian_mid_aligned[i]
        
        # Breakout conditions
        breakout_long = price_12h > dh
        breakout_short = price_12h < dl
        
        # Trend filter: 1d close vs 1w EMA50
        close_1d_current = close_1d[i]
        ema_50_1w_current = ema_50_1w_aligned[i]
        bullish_trend = close_1d_current > ema_50_1w_current
        bearish_trend = close_1d_current < ema_50_1w_current
        
        # Exit conditions
        exit_long = price_12h < dm  # Price crosses below midpoint
        exit_short = price_12h > dm  # Price crosses above midpoint
        
        # Entry conditions
        enter_long = breakout_long and bullish_trend and vol_confirm
        enter_short = breakout_short and bearish_trend and vol_confirm
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals