#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR-based trend filter
# - Long: price breaks above 20-period Donchian high with volume > 1.5x 20-period average and price > 50-period EMA
# - Short: price breaks below 20-period Donchian low with volume > 1.5x 20-period average and price < 50-period EMA
# - Exit: price returns to opposite Donchian level (mean reversion at channel midpoint)
# - Uses 1d HTF trend filter: only take longs when price > 1d 200-period EMA, shorts when price < 1d 200-period EMA
# - Target: 20-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by combining breakout momentum with trend filtering

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for HTF trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return signals
    
    # Pre-compute 1d 200-period EMA for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 4h 50-period EMA for additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Donchian levels
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        
        # Trend filters
        price_above_50ema = close_price > ema_50[i]
        price_below_50ema = close_price < ema_50[i]
        price_above_1d_200ema = close_price > ema_200_1d_aligned[i]
        price_below_1d_200ema = close_price < ema_200_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above Donchian high with volume and EMA filters
        if close_price > upper_channel and vol_confirm and price_above_50ema and price_above_1d_200ema:
            enter_long = True
        
        # Short breakout: price breaks below Donchian low with volume and EMA filters
        if close_price < lower_channel and vol_confirm and price_below_50ema and price_below_1d_200ema:
            enter_short = True
        
        # Exit conditions: mean reversion at opposite Donchian level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to Donchian low
            exit_long = close_price <= lower_channel
        elif position == -1:
            # Exit short if price rises back to Donchian high
            exit_short = close_price >= upper_channel
        
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