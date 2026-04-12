#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extremes with 12h trend filter and volume confirmation
    # Williams %R identifies overbought/oversold conditions for mean reversion
    # 12h EMA50 provides trend direction to avoid counter-trend whipsaws
    # Volume spike confirms institutional participation at extremes
    # Works in bull/bear: fade extremes in range, follow trend in strong markets
    # Target: 12-30 trades/year per symbol (50-120 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Williams %R (14-period) on 12h
    williams_r = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):
        highest_high = np.max(high_12h[i-13:i+1])
        lowest_low = np.min(low_12h[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_12h[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # 12h EMA50 for trend filter
    if len(close_12h) >= 50:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_12h = np.full(len(df_12h), np.nan)
    
    # 12h volume spike filter (current volume > 2.0 * 20-period average)
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    for i in range(19, len(df_12h)):
        vol_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    volume_spike = volume > 2.0 * vol_ma_20_12h_aligned
    
    # Align 12h indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extremes: >80 = overbought, <20 = oversold
        # Trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        
        # Mean reversion logic at extremes
        long_entry = williams_r_aligned[i] < -80 and close[i] > ema_50_12h_aligned[i] and volume_spike[i]
        short_entry = williams_r_aligned[i] > -20 and close[i] < ema_50_12h_aligned[i] and volume_spike[i]
        
        # Exit when Williams %R returns to neutral range (-50 to -50) or volume drops
        long_exit = williams_r_aligned[i] > -50 or (not volume_spike[i])
        short_exit = williams_r_aligned[i] < -50 or (not volume_spike[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_williamsr_extreme_trend_vol_v1"
timeframe = "6h"
leverage = 1.0