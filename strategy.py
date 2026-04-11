#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Daily Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Volume confirmation: 20-period average on daily volume
    volume_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20)
    
    # Trend filter: 50-period EMA on daily close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period daily average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Trend filter: only trade in direction of 50 EMA
        above_ema = price_close > ema_50_aligned[i]
        below_ema = price_close < ema_50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Donchian high + volume confirmation + above EMA50
        if price_close > donchian_high_20_aligned[i] and vol_confirm and above_ema:
            enter_long = True
        
        # Short: Price breaks below Donchian low + volume confirmation + below EMA50
        if price_close < donchian_low_20_aligned[i] and vol_confirm and below_ema:
            enter_short = True
        
        # Exit conditions: price returns to the day's midpoint
        donchian_mid = (donchian_high_20_aligned[i] + donchian_low_20_aligned[i]) / 2
        exit_long = price_close < donchian_mid
        exit_short = price_close > donchian_mid
        
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

# Hypothesis: Donchian breakout on daily timeframe with volume confirmation and EMA50 trend filter.
# Works in both bull (breakouts above Donchian high) and bear (breakdowns below Donchian low) by capturing institutional activity.
# Volume confirmation (>1.5x 20-day average) ensures participation. EMA50 filter avoids counter-trend trades.
# Position size 0.25 balances risk and return. Target: 15-30 trades/year to minimize fee drag.
# Uses 4h primary timeframe for execution, with 1d for signal generation to reduce noise.