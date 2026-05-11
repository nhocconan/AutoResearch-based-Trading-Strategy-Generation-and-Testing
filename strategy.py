#!/usr/bin/env python3
"""
1d_Weekly_Price_Channel_Strategy_v1
Hypothesis: Uses weekly Donchian channels to identify long-term trends and daily price action for entry timing.
In bull markets, buys near weekly support with upward bias. In bear markets, sells near weekly resistance with downward bias.
Uses volume confirmation and volatility filter to avoid false breaks. Designed for low frequency (10-25 trades/year) 
to work across market regimes by aligning with weekly structure while using daily precision.
"""

name = "1d_Weekly_Price_Channel_Strategy_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend context
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Daily OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-week lookback)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_high_max = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_low_min = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly channels to daily
    weekly_high_max_daily = align_htf_to_ltf(prices, df_weekly, weekly_high_max)
    weekly_low_min_daily = align_htf_to_ltf(prices, df_weekly, weekly_low_min)
    
    # Weekly trend filter (50-week EMA)
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_daily = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Daily volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    # Daily volatility filter (ATR-based)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_max_daily[i]) or 
            np.isnan(weekly_low_min_daily[i]) or
            np.isnan(weekly_ema50_daily[i]) or
            np.isnan(atr[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: price near weekly support with bullish bias
        near_weekly_support = low[i] <= weekly_low_min_daily[i] * 1.002  # Within 0.2% of weekly low
        bullish_bias = close[i] > weekly_ema50_daily[i]  # Above weekly EMA50
        vol_confirm = vol_spike[i]
        volatility_ok = atr[i] > 0  # Ensure volatility exists
        
        long_signal = near_weekly_support and bullish_bias and vol_confirm and volatility_ok
        
        # Short conditions: price near weekly resistance with bearish bias
        near_weekly_resistance = high[i] >= weekly_high_max_daily[i] * 0.998  # Within 0.2% of weekly high
        bearish_bias = close[i] < weekly_ema50_daily[i]  # Below weekly EMA50
        short_signal = near_weekly_resistance and bearish_bias and vol_confirm and volatility_ok
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: opposite signal or price moves to middle of channel
            if position == 1:
                # Exit long if price reaches midpoint of weekly channel or gets opposite signal
                weekly_mid = (weekly_high_max_daily[i] + weekly_low_min_daily[i]) / 2
                exit_signal = short_signal or (close[i] >= weekly_mid)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price reaches midpoint of weekly channel or gets opposite signal
                weekly_mid = (weekly_high_max_daily[i] + weekly_low_min_daily[i]) / 2
                exit_signal = long_signal or (close[i] <= weekly_mid)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals