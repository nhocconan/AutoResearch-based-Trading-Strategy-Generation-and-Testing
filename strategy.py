#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h ADX regime filter
# - Long: Close breaks above Donchian upper (20) AND volume > 1.5x 20-period average AND 12h ADX > 25
# - Short: Close breaks below Donchian lower (20) AND volume > 1.5x 20-period average AND 12h ADX > 25
# - Exit: Close crosses opposite Donchian band OR volume drops below average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian channels provide clear structure for breakouts
# - Volume confirmation ensures participation
# - 12h ADX > 25 filters for trending regimes only (avoids chop)
# - Works in both bull (strong uptrends) and bear (strong downtrends) markets

name = "4h_12h_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # +DM and -DM
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if period < len(data):
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilder_smooth(tr, 14)
    plus_di_12h = 100 * wilder_smooth(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilder_smooth(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilder_smooth(dx_12h, 14)
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = close_current > donchian_upper[i-1]  # Use previous bar's upper band
        breakout_down = close_current < donchian_lower[i-1]  # Use previous bar's lower band
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Regime filter: 12h ADX > 25 (trending market)
        trending = adx_12h_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish breakout + volume confirmation + trending regime
        if breakout_up and vol_confirm and trending:
            enter_long = True
        
        # Short: bearish breakout + volume confirmation + trending regime
        if breakout_down and vol_confirm and trending:
            enter_short = True
        
        # Exit conditions: opposite breakout OR loss of volume confirmation OR non-trending
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bearish breakout OR volume drops below average OR market becomes choppy
            exit_long = (breakout_down) or (not vol_confirm) or (not trending)
        elif position == -1:
            # Exit short if bullish breakout OR volume drops below average OR market becomes choppy
            exit_short = (breakout_up) or (not vol_confirm) or (not trending)
        
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