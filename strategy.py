#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h trend filter + volume confirmation
# - Donchian breakout on 6h: price > 20-period high for long, price < 20-period low for short
# - Trend filter: 12h EMA50 > EMA200 for long bias, EMA50 < EMA200 for short bias
# - Volume confirmation: 6h volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves
# - 12h EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation filters out weak breakouts
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets

name = "6h_12h_donchian_volume_trend_v1"
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Pre-compute 12h trend bias (1 for uptrend, -1 for downtrend, 0 for neutral)
    trend_bias = np.zeros(len(ema_50_aligned))
    trend_bias[ema_50_aligned > ema_200_aligned] = 1
    trend_bias[ema_50_aligned < ema_200_aligned] = -1
    
    # Pre-compute 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(trend_bias[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = price_close > donchian_high[i-1]  # Close above previous period's high
        breakout_short = price_close < donchian_low[i-1]  # Close below previous period's low
        
        # Trend filter from 12h
        trend_up = trend_bias[i] == 1
        trend_down = trend_bias[i] == -1
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout + uptrend + volume confirmation
        if breakout_long and trend_up and vol_confirm:
            enter_long = True
        
        # Short: Donchian breakdown + downtrend + volume confirmation
        if breakout_short and trend_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Donchian breakdown OR trend turns down
            exit_long = (price_close < donchian_low[i-1]) or (not trend_up)
        elif position == -1:
            # Exit short if Donchian breakout OR trend turns up
            exit_short = (price_close > donchian_high[i-1]) or (not trend_down)
        
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