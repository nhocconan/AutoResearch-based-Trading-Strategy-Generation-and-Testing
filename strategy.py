#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (bull power/bear power) with 12-hour trend filter and 1-day volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Trend filter uses EMA34 on 12h to avoid counter-trend trades. Volume confirms institutional participation.
# Designed for 6h timeframe to target 50-150 trades over 4 years with low frequency.
# Works in bull/bear via trend alignment and mean-reversion at extremes.

name = "6h_elder_ray12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6-hour EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12-hour EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=5, min_periods=1).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 34  # EMA34 needs 34 periods
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Trend filter: bullish if price > EMA34, bearish if price < EMA34
        bullish_trend = close[i] > ema34_12h_aligned[i]
        bearish_trend = close[i] < ema34_12h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bear power > 0 (selling pressure) or stoploss
            if (bear_power[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power < 0 (no buying pressure) or stoploss
            if (bull_power[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: bull power > 0 and bearish trend (mean reversion in uptrend)
                if bull_power[i] > 0 and bearish_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bear power < 0 and bullish trend (mean reversion in downtrend)
                elif bear_power[i] < 0 and bullish_trend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals