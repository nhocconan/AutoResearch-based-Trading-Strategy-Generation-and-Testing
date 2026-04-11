#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA filter + volume confirmation
# - Long when price breaks above Donchian(20) high + price > 1d EMA200 + volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low + price < 1d EMA200 + volume > 1.5x 20-period average
# - Exit when price reverses to Donchian midpoint or opposite breakout occurs
# - Uses discrete position sizing ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total) to stay within fee drag limits for 4h
# - Works in bull markets (trend continuation) and bear markets (trend reversal on breakdowns)
# - 1d EMA200 provides strong trend filter, reducing false signals in choppy markets

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return signals
    
    # Pre-compute 1d EMA200
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA200 trend filter
        price_above_ema200 = price_close > ema200_1d_aligned[i]
        price_below_ema200 = price_close < ema200_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above Donchian high + price above 1d EMA200 + volume confirmation
        if price_close > donchian_high[i] and price_above_ema200 and vol_confirm:
            enter_long = True
        
        # Short: Break below Donchian low + price below 1d EMA200 + volume confirmation
        if price_close < donchian_low[i] and price_below_ema200 and vol_confirm:
            enter_short = True
        
        # Exit conditions: price reverses to midpoint or opposite breakout
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Donchian midpoint or breaks below Donchian low
            exit_long = (price_close <= donchian_mid[i]) or (price_close < donchian_low[i])
        elif position == -1:
            # Exit short if price returns to Donchian midpoint or breaks above Donchian high
            exit_short = (price_close >= donchian_mid[i]) or (price_close > donchian_high[i])
        
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