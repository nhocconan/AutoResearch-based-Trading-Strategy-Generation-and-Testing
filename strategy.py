#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly trend filter + volume confirmation
# - Donchian(20) breakout: long when price breaks above 20-period high, short when breaks below 20-period low
# - Weekly trend filter: price above/below weekly EMA50 to align with higher timeframe trend
# - Volume confirmation: current volume > 1.5x 20-period average to filter false breakouts
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in both bull (trend continuation with volume) and bear (trend reversal with volume) markets
# - Weekly EMA50 provides strong trend filter, reducing false signals in choppy markets

name = "6h_1w_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute weekly EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = price_high > donchian_high[i]  # Price breaks above 20-period high
        breakout_down = price_low < donchian_low[i]  # Price breaks below 20-period low
        
        # Weekly trend filter
        price_above_weekly_ema50 = price_close > ema50_1w_aligned[i]
        price_below_weekly_ema50 = price_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Upward breakout + price above weekly EMA50 + volume confirmation
        if breakout_up and price_above_weekly_ema50 and vol_confirm:
            enter_long = True
        
        # Short: Downward breakout + price below weekly EMA50 + volume confirmation
        if breakout_down and price_below_weekly_ema50 and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite breakout or price crosses weekly EMA50
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if downward breakout OR price crosses below weekly EMA50
            exit_long = breakout_down or (not price_above_weekly_ema50)
        elif position == -1:
            # Exit short if upward breakout OR price crosses above weekly EMA50
            exit_short = breakout_up or (not price_below_weekly_ema50)
        
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