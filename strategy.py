#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 1-day trend filter and 1-week volume confirmation
# Elder Ray measures bull/bear power relative to EMA. Long when bull power > 0 and rising, short when bear power < 0 and falling.
# Uses 1-day EMA for trend direction to avoid counter-trend trades, and 1-week volume average for confirmation.
# Designed to work in both bull (captures rallies) and bear (avoids false breaks) markets.
# Target: 75-150 total trades over 4 years (19-38/year)

name = "6h_elder_ray_1d_trend_1w_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter (EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-week data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    # 1-day EMA for trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1-week volume average (20-period)
    volume_1w = df_1w['volume'].values
    volume_1w_s = pd.Series(volume_1w)
    volume_ma_1w = volume_1w_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: bear power becomes positive (momentum fading)
            elif bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: bull power becomes negative (momentum fading)
            elif bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals with trend and volume filters
            # Trend filter: price above/below 1-day EMA
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            # Volume filter: current volume > 1.5x weekly average
            volume_filter = volume[i] > 1.5 * volume_ma_1w_aligned[i]
            
            # Long: bull power positive AND rising (momentum building) + uptrend + volume
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bear power negative AND falling (momentum building) + downtrend + volume
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals