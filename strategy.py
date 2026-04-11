#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with volume confirmation and weekly trend filter
# - Long: Close > Donchian Upper(20) on 1d AND volume > 1.5x 20-day average AND weekly close > weekly EMA20
# - Short: Close < Donchian Lower(20) on 1d AND volume > 1.5x 20-day average AND weekly close < weekly EMA20
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# - Donchian channels identify clear breakouts with structure
# - Volume confirmation ensures breakout validity
# - Weekly EMA filter aligns with higher timeframe trend to avoid counter-trend trades
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets

name = "1d_donchian_volume_weekly_trend_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Pre-compute 1d Donchian channels (20-period)
    # Upper band: 20-period high
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(weekly_ema20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = close_current > donchian_upper[i]
        breakdown_dn = close_current < donchian_lower[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Weekly trend filter: price above/below weekly EMA20
        weekly_uptrend = close_current > weekly_ema20_aligned[i]
        weekly_downtrend = close_current < weekly_ema20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + volume confirmation + weekly uptrend
        if breakout_up and vol_confirm and weekly_uptrend:
            enter_long = True
        
        # Short: Donchian breakdown down + volume confirmation + weekly downtrend
        if breakdown_dn and vol_confirm and weekly_downtrend:
            enter_short = True
        
        # Exit conditions: opposite Donchian touch or loss of volume confirmation
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches Donchian lower band OR volume confirmation lost
            exit_long = (close_current <= donchian_lower[i]) or (not vol_confirm)
        elif position == -1:
            # Exit short if price touches Donchian upper band OR volume confirmation lost
            exit_short = (close_current >= donchian_upper[i]) or (not vol_confirm)
        
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