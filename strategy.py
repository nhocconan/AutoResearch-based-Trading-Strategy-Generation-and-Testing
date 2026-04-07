#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with 1d Trend and Volume
# Hypothesis: 12h Donchian(20) breakouts capture medium-term trends. 
# 1d EMA50 > SMA50 filter ensures trades align with higher timeframe trend.
# Volume confirmation (1.5x 20-period average) filters low-conviction moves.
# Works in bull (breakouts above resistance) and bear (breakdowns below support).
# Target: 15-35 trades/year (60-140 over 4 years) to stay under 200 total trades limit.
name = "12h_donchian20_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 1d trend filter: EMA50 > SMA50 = bullish
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_sma50 = pd.Series(daily_close).rolling(window=50, min_periods=50).mean().values
    daily_ema50_12h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    daily_sma50_12h = align_htf_to_ltf(prices, df_1d, daily_sma50)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(daily_ema50_12h[i]) or np.isnan(daily_sma50_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from daily data
        bull_regime = daily_ema50_12h[i] > daily_sma50_12h[i]  # Bullish when EMA > SMA
        bear_regime = daily_ema50_12h[i] < daily_sma50_12h[i]  # Bearish when EMA < SMA
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend reversal
            if close[i] <= donchian_low[i] or not bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend reversal
            if close[i] >= donchian_high[i] or not bear_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Bull regime: long on breakout above Donchian high
                if bull_regime and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Bear regime: short on breakdown below Donchian low
                elif bear_regime and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals