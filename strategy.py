#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# - Long: price breaks above Donchian(20) high, volume > 1.5x 20-period avg, weekly EMA(21) > weekly EMA(50) (bullish trend)
# - Short: price breaks below Donchian(20) low, volume > 1.5x 20-period avg, weekly EMA(21) < weekly EMA(50) (bearish trend)
# - Exit: price returns to Donchian midpoint
# - Uses 1d timeframe with 1w HTF for trend alignment
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by aligning breakouts with weekly trend direction

name = "1d_1w_donchian_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for weekly EMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(21) and EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        mid_channel = donchian_mid[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Weekly trend filter: EMA(21) > EMA(50) for bullish, < for bearish
        weekly_bullish = ema_21_1w_aligned[i] > ema_50_1w_aligned[i]
        weekly_bearish = ema_21_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above upper Donchian, volume confirmation, weekly bullish trend
        if close_price > upper_channel and vol_confirm and weekly_bullish:
            enter_long = True
        
        # Short breakout: price below lower Donchian, volume confirmation, weekly bearish trend
        if close_price < lower_channel and vol_confirm and weekly_bearish:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Donchian midpoint
            exit_long = close_price <= mid_channel
        elif position == -1:
            # Exit short if price returns to Donchian midpoint
            exit_short = close_price >= mid_channel
        
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