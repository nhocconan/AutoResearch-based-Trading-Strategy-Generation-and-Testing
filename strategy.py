#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly trend filter + volume confirmation
# - Primary: 6h Donchian breakout (20-period) for entry timing
# - HTF trend: Weekly close above/below 200-EMA for trend direction
# - Volume: 6h volume > 1.5x 20-period average for confirmation
# - Long: Price breaks above Donchian Upper AND weekly trend bullish AND volume confirmation
# - Short: Price breaks below Donchian Lower AND weekly trend bearish AND volume confirmation
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Donchian provides objective breakout levels
# - Weekly 200-EMA filter ensures trading with higher timeframe trend
# - Volume confirmation filters out weak breakouts
# - Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume)

name = "6h_1w_donchian_weekly_trend_volume_v1"
timeframe = "6h"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute weekly 200-EMA for trend filter
    weekly_close = df_1w['close'].values
    ema200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_trend_bullish = weekly_close > ema200_1w
    weekly_trend_bearish = weekly_close < ema200_1w
    weekly_trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bullish.astype(float))
    weekly_trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bearish.astype(float))
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 6h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):  # Start after warmup for Donchian and EMA
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(weekly_trend_bullish_aligned[i]) or
            np.isnan(weekly_trend_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = close_current > donchian_upper[i-1]  # Break above upper channel
        breakdown_down = close_current < donchian_lower[i-1]  # Break below lower channel
        
        # Weekly trend filter
        trend_bullish = weekly_trend_bullish_aligned[i] > 0.5
        trend_bearish = weekly_trend_bearish_aligned[i] > 0.5
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + weekly bullish trend + volume confirmation
        if breakout_up and trend_bullish and vol_confirm:
            enter_long = True
        
        # Short: Donchian breakdown down + weekly bearish trend + volume confirmation
        if breakdown_down and trend_bearish and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or loss of trend
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Donchian breakdown OR weekly trend turns bearish
            exit_long = breakdown_down or (not trend_bullish)
        elif position == -1:
            # Exit short if Donchian breakout up OR weekly trend turns bullish
            exit_short = breakout_up or (not trend_bearish)
        
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