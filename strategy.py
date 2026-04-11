#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + volume confirmation + weekly trend filter
# - Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + weekly EMA50 rising
# - Short when price breaks below Donchian(20) low + volume > 1.5x 20-period average + weekly EMA50 falling
# - Uses discrete position sizing: ±0.30 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Works in both bull (breakouts with volume in uptrend) and bear (breakdowns with volume in downtrend) markets
# - Weekly EMA50 provides strong trend filter, reducing false signals in choppy markets

name = "12h_1w_donchian_volume_trend_v1"
timeframe = "12h"
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
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = price_close > donchian_high[i-1]  # Break above previous high
        breakdown_down = price_close < donchian_low[i-1]  # Break below previous low
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Weekly EMA50 trend filter: rising/falling
        ema50_prev = ema50_1w_aligned[i-1] if i > 0 else ema50_1w_aligned[i]
        ema50_rising = ema50_1w_aligned[i] > ema50_prev
        ema50_falling = ema50_1w_aligned[i] < ema50_prev
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + volume confirmation + weekly EMA50 rising
        if breakout_up and vol_confirm and ema50_rising:
            enter_long = True
        
        # Short: Donchian breakdown down + volume confirmation + weekly EMA50 falling
        if breakdown_down and vol_confirm and ema50_falling:
            enter_short = True
        
        # Exit conditions: opposite breakout or loss of volume confirmation
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if breakdown OR loss of volume confirmation
            exit_long = breakdown_down or not vol_confirm
        elif position == -1:
            # Exit short if breakout OR loss of volume confirmation
            exit_short = breakout_up or not vol_confirm
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals