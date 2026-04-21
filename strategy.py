#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high and weekly trend is up (price > weekly EMA20).
# Short when price breaks below 20-day low and weekly trend is down (price < weekly EMA20).
# Uses volume > 1.5x 20-day average for confirmation to avoid false breakouts.
# Weekly EMA provides trend filter to avoid whipsaws in sideways markets.
# Target: 15-25 trades/year by requiring trend alignment + breakout + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly EMA20 on close
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Align to daily timeframe (will be available after weekly bar closes)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute daily indicators
    # Daily volume 20-period moving average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period) on daily data
        lookback_start = max(0, i - 19)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        donchian_high = np.max(high_window)
        donchian_low = np.min(low_window)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Weekly trend filter
        weekly_trend_up = price > ema_20_1w_aligned[i]
        weekly_trend_down = price < ema_20_1w_aligned[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            if weekly_trend_up and volume_confirm:
                # Uptrend: look for upside breakout
                if price > donchian_high:
                    signals[i] = 0.25
                    position = 1
            elif weekly_trend_down and volume_confirm:
                # Downtrend: look for downside breakout
                if price < donchian_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: opposite breakout or loss of trend
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on downside breakout or if trend turns down
                if price < donchian_low or not weekly_trend_up:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on upside breakout or if trend turns up
                if price > donchian_high or not weekly_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_DonchianBreakout_WeeklyEMA20Trend_Volume"
timeframe = "1d"
leverage = 1.0