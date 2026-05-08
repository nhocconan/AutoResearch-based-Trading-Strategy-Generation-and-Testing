#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Trend + Daily Volume Breakout with Volatility Filter
# Uses weekly EMA20 for long-term trend direction, daily volume > 2x 20-period average for entry,
# and ATR-based volatility filter to avoid choppy markets. Designed to capture strong trends
# in both bull and bear markets by following weekly momentum while requiring volume confirmation.
# Target: 15-25 trades/year on daily timeframe.

name = "1d_WeeklyTrend_VolumeBreakout_VolatilityFilter"
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
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema20_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 20:
        ema20_weekly[19] = np.mean(close_weekly[:20])
        for i in range(20, len(close_weekly)):
            ema20_weekly[i] = (close_weekly[i] * 2 + ema20_weekly[i-1] * 18) / 20
    
    # Calculate daily ATR(14) for volatility filter
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.nanmean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily volume average for volume breakout
    vol_avg_20 = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly indicators to daily timeframe
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema20_weekly_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely low volatility (choppy) markets
        # Use ATR ratio to current price - only trade when volatility is sufficient
        vol_filter = atr_14[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Volume breakout: current daily volume > 2x 20-period average
        vol_breakout = volume[i] > 2.0 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: follow weekly EMA trend with volume breakout and sufficient volatility
            long_condition = (
                close[i] > ema20_weekly_aligned[i] and   # price above weekly EMA20 (bullish bias)
                vol_breakout and                         # volume breakout for entry
                vol_filter                               # sufficient volatility
            )
            
            short_condition = (
                close[i] < ema20_weekly_aligned[i] and   # price below weekly EMA20 (bearish bias)
                vol_breakout and                         # volume breakout for entry
                vol_filter                               # sufficient volatility
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below weekly EMA20 or volatility drops
            if close[i] < ema20_weekly_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above weekly EMA20 or volatility drops
            if close[i] > ema20_weekly_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals