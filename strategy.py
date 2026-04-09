#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v2
# Hypothesis: Daily Camarilla pivot levels with weekly trend filter and volume confirmation.
# Long: Price touches Camarilla L3 support, weekly close > weekly open (bullish week), volume > 1.5x 20-day average.
# Short: Price touches Camarilla H3 resistance, weekly close < weekly open (bearish week), volume > 1.5x 20-day average.
# Exit: Opposite Camarilla level touch (L4/H4) or ATR trailing stop (2.5x ATR from extreme).
# Uses weekly trend for direction, Camarilla for precise entries/exits, volume for confirmation.
# Target: 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly bullish/bearish: close > open for bullish week
    close_1w = pd.Series(df_1w['close'].values)
    open_1w = pd.Series(df_1w['open'].values)
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align HTF weekly trend to daily timeframe (wait for completed 1w bar)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.values.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.values.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(open_price[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price touches or exceeds Camarilla H4 level
            elif close[i] >= camarilla_h4[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if short_low > 0 and close[i] > short_low + 2.5 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Price touches or exceeds Camarilla L4 level
            elif close[i] <= camarilla_l4[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate Camarilla levels for today using yesterday's OHLC
            if i >= 1:
                # Camarilla levels: based on previous day's range
                phigh = high[i-1]
                plow = low[i-1]
                pclose = close[i-1]
                range_val = phigh - plow
                
                camarilla_h3 = pclose + range_val * 1.1 / 4
                camarilla_l3 = pclose - range_val * 1.1 / 4
                camarilla_h4 = pclose + range_val * 1.1 / 2
                camarilla_l4 = pclose - range_val * 1.1 / 2
            else:
                camarilla_h3 = camarilla_l3 = camarilla_h4 = camarilla_l4 = 0.0
            
            # Long entry: Price touches L3 support, weekly bullish, volume confirmed
            if (low[i] <= camarilla_l3 and 
                weekly_bullish_aligned[i] > 0.5 and 
                volume_confirmed):
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: Price touches H3 resistance, weekly bearish, volume confirmed
            elif (high[i] >= camarilla_h3 and 
                  weekly_bearish_aligned[i] > 0.5 and 
                  volume_confirmed):
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals