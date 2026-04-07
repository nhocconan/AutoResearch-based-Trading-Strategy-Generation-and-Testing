#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d ADX Trend with Weekly Volume Confirmation
# Hypothesis: ADX > 25 indicates strong trend, +DI > -DI for long, -DI > +DI for short.
# Weekly volume > 1.5x average confirms institutional participation. Works in bull/bear.
# Daily timeframe targets 7-25 trades/year. Uses discrete position sizing to minimize fees.
name = "1d_adx_trend_weekly_volume_v1"
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
    volume = prices['volume'].values
    
    # Get weekly data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Weekly volume filter: current week volume > 1.5x 4-week average
    weekly_volume = df_1w['volume'].values
    vol_ma = pd.Series(weekly_volume).rolling(window=4, min_periods=4).mean().values
    vol_filter = weekly_volume > (vol_ma * 1.5)
    vol_filter_1d = align_htf_to_ltf(prices, df_1w, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(vol_filter_1d[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        strong_trend = adx[i] > 25
        bullish = plus_di[i] > minus_di[i]
        bearish = minus_di[i] > plus_di[i]
        
        if position == 1:  # Long position
            # Exit: trend weakens or reverses
            if not strong_trend or not bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: trend weakens or reverses
            if not strong_trend or not bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require strong trend and volume confirmation
            if strong_trend and vol_filter_1d[i]:
                if bullish:
                    position = 1
                    signals[i] = 0.25
                elif bearish:
                    position = -1
                    signals[i] = -0.25
    
    return signals