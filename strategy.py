#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_v1
# Hypothesis: 12h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long: Price breaks above 20-period Donchian high, volume > 1.5x 20-period average, and chop < 61.8 (trending regime).
# Short: Price breaks below 20-period Donchian low, volume > 1.5x 20-period average, and chop < 61.8 (trending regime).
# Exit: Opposite Donchian breakout or ATR trailing stop (2.5x ATR from extreme).
# Uses 1w EMA as higher timeframe trend filter to avoid counter-trend trades.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    
    # Donchian channel (20-period)
    highest_high = high_s.rolling(window=20, min_periods=20).max().values
    lowest_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period) for regime filter
    chop_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high_14 = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low_14 = low_s.rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop = np.where(chop_denom != 0, 100 * np.log10(chop_sum / chop_denom) / np.log10(14), 50.0)
    
    # Get 1w data for EMA trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF EMA to 12h timeframe (wait for completed 1w bar)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(close[i]) or
            np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: price above/below weekly EMA20
        price_above_weekly_ema = close[i] > ema20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema20_1w_aligned[i]
        
        # Choppiness regime filter: chop < 61.8 indicates trending market (good for breakouts)
        trending_regime = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below Donchian low (20-period)
            elif close[i] < lowest_low[i]:
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
            # Exit: Price breaks above Donchian high (20-period)
            elif close[i] > highest_high[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high, volume confirmed, above weekly EMA, trending regime
            if (close[i] > highest_high[i] and volume_confirmed and price_above_weekly_ema and trending_regime):
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, volume confirmed, below weekly EMA, trending regime
            elif (close[i] < lowest_low[i] and volume_confirmed and price_below_weekly_ema and trending_regime):
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals