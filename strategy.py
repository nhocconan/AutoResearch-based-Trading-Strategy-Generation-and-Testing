#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout with 1-week Trend Filter
# Hypothesis: Buy breakouts above 20-day high in bullish weekly trend, sell breakdowns below 20-day low in bearish weekly trend.
# Weekly trend defined by price above/below 20-week EMA. This captures medium-term momentum while avoiding counter-trend trades.
# Volatility filter: Only trade when ATR(14) > 20-period average ATR to avoid low-volatility chop.
# Position size: 0.25 for clear breakouts with confirmation.
# Target: 15-25 trades/year (60-100 over 4 years) - low frequency reduces fee drag.
name = "1d_donchian_breakout_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on daily
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Weekly EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    atr_ma = atr.rolling(window=20, min_periods=20).mean()
    vol_filter = atr > atr_ma  # High volatility regime
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_ema_1d[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 20-day low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above 20-day high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require high volatility regime
            if vol_filter[i]:
                # Bullish weekly trend: look for long breakouts
                if close[i] > weekly_ema_1d[i]:
                    # Long: price breaks above 20-day high
                    if close[i] > donchian_high[i]:
                        position = 1
                        signals[i] = 0.25
                # Bearish weekly trend: look for short breakdowns
                elif close[i] < weekly_ema_1d[i]:
                    # Short: price breaks below 20-day low
                    if close[i] < donchian_low[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals