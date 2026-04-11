#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation
# - Long when price breaks above Donchian high (20-day) + price > 1w EMA200 + volume > 1.5x 20-day avg
# - Short when price breaks below Donchian low (20-day) + price < 1w EMA200 + volume > 1.5x 20-day avg
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits for 1d
# - Works in both bull (trend continuation with volume) and bear (trend reversal with volume) markets
# - 1w EMA200 provides strong trend filter, reducing false signals in choppy markets
# - Volume confirms institutional participation in breakouts

name = "1d_1w_donchian_ema_volume_trend_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return signals
    
    # Pre-compute 1w EMA200
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute Donchian channels (20-day)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume SMA (20-day)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = price_close > donchian_high[i]  # New 20-day high
        breakout_down = price_close < donchian_low[i]  # New 20-day low
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1w EMA200 trend filter
        price_above_ema200 = price_close > ema200_1w_aligned[i]
        price_below_ema200 = price_close < ema200_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + price above 1w EMA200 + volume confirmation
        if breakout_up and price_above_ema200 and vol_confirm:
            enter_long = True
        
        # Short: Donchian breakout down + price below 1w EMA200 + volume confirmation
        if breakout_down and price_below_ema200 and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or price crosses 1w EMA200
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Donchian breakout down OR price crosses below 1w EMA200
            exit_long = breakout_down or (not price_above_ema200)
        elif position == -1:
            # Exit short if Donchian breakout up OR price crosses above 1w EMA200
            exit_short = breakout_up or (not price_below_ema200)
        
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