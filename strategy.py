#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d trend filter (EMA50 > EMA200) + volume confirmation
# - Donchian breakout on 4h: price > 20-period high for long, price < 20-period low for short
# - Trend filter: 1d EMA50 > EMA200 for long bias, EMA50 < EMA200 for short bias
# - Volume confirmation: 4h volume > 1.8x 20-period average (stricter to reduce trades)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - 1d EMA filter ensures we trade with the higher timeframe trend (more reliable than 12h)
# - Volume confirmation filters out weak breakouts, reducing false signals
# - Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume)

name = "4h_1d_donchian_volume_trend_v2"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 1d trend bias (1 for uptrend, -1 for downtrend, 0 for neutral)
    trend_bias = np.zeros(len(ema_50_aligned))
    trend_bias[ema_50_aligned > ema_200_aligned] = 1
    trend_bias[ema_50_aligned < ema_200_aligned] = -1
    
    # Pre-compute 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation (20-period average)
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
        
        # Trend filter from 1d
        trend_up = trend_bias[i] == 1
        trend_down = trend_bias[i] == -1
        
        # Volume confirmation: current volume > 1.8x 20-period average (stricter)
        vol_confirm = volume_current > 1.8 * volume_sma_20[i]
        
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