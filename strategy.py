#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v4
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long: Price breaks above Donchian(20) high, volume > 1.5x 20-period average, and CHOP(14) > 61.8 (range regime for mean reversion).
# Short: Price breaks below Donchian(20) low, volume > 1.5x 20-period average, and CHOP(14) > 61.8.
# Exit: Opposite Donchian break or ATR trailing stop (2.5x ATR from extreme).
# Uses daily trend filter: only take longs when price > daily EMA50, shorts when price < daily EMA50.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v4"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    high_s_rolled = pd.Series(high)
    low_s_rolled = pd.Series(low)
    donchian_high = high_s_rolled.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s_rolled.rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    chop_period = 14
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    high_max = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    low_min = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop_denom = np.log10(high_max - low_min) * np.sqrt(chop_period)
    chop = 100 * np.log10(atr_sum / chop_denom) / np.log10(chop_period)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness filter: only trade in range regimes (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        # Daily trend filter: only long when price > daily EMA50, short when price < daily EMA50
        trend_filter_long = close[i] > ema_50_1d_aligned[i]
        trend_filter_short = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below Donchian low
            elif low[i] < donchian_low[i]:
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
            # Exit: Price breaks above Donchian high
            elif high[i] > donchian_high[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high, volume confirmed, chop regime, and uptrend
            if (high[i] > donchian_high[i] and volume_confirmed and chop_filter and trend_filter_long):
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, volume confirmed, chop regime, and downtrend
            elif (low[i] < donchian_low[i] and volume_confirmed and chop_filter and trend_filter_short):
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals