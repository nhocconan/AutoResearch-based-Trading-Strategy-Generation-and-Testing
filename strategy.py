#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour VWAP bounce with 4-hour trend filter and daily volume confirmation
# Long when price > VWAP + price > 4h EMA20 + volume > 1.2x daily average
# Short when price < VWAP + price < 4h EMA20 + volume > 1.2x daily average
# Exit when price crosses VWAP in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h EMA for trend direction and daily volume for confirmation
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_vwap_bounce_4h_ema_daily_vol_v1"
timeframe = "1h"
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
    
    # 4-hour data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(20)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / (vwap_denominator + 1e-10)
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below VWAP
            elif close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above VWAP
            elif close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: VWAP bounce with 4h EMA trend and volume confirmation
            # Trend filter: price > 4h EMA20 for long, price < 4h EMA20 for short
            trend_filter_long = close[i] > ema_4h_aligned[i]
            trend_filter_short = close[i] < ema_4h_aligned[i]
            # Volume filter: volume > 1.2x daily average
            volume_filter = volume[i] > 1.2 * volume_ma_aligned[i]
            
            # Long: price > VWAP + price > 4h EMA20 + volume filter
            if close[i] > vwap[i] and trend_filter_long and volume_filter:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price < VWAP + price < 4h EMA20 + volume filter
            elif close[i] < vwap[i] and trend_filter_short and volume_filter:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals