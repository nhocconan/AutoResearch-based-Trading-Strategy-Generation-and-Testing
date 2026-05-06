#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Breakout with 1w EMA200 trend filter and volume spike
# Uses 1d Bollinger Bands (20,2) for breakout signals, 1w EMA200 for trend alignment (reduces whipsaw)
# Volume spike (>2.0x 20-bar average) confirms breakout strength
# ATR-based trailing stop via signal=0 when price retraces 20% of ATR from extreme
# Discrete sizing 0.25 to balance profit potential and fee drag; target 40-80 total trades over 4 years (10-20/year)
# Works in both bull/bear: breakouts capture momentum, trend filter avoids counter-trend traps, volume filter ensures participation
# Bollinger Bands provide dynamic support/resistance that adapts to volatility, effective in ranging and trending markets

name = "1d_BollingerBreakout_1wEMA200_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Align HTF indicators to 1d timeframe (primary)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute session filter (00-23 UTC for daily timeframe)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = np.ones(n, dtype=bool)  # Always active for daily timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > upper band AND uptrend (price > EMA200) AND volume spike
            if close[i] > upper_band[i] and close[i] > ema200_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short breakdown: price < lower band AND downtrend (price < EMA200) AND volume spike
            elif close[i] < lower_band[i] and close[i] < ema200_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 20% of ATR from extreme
            if close[i] <= long_extreme - 0.20 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 20% of ATR from extreme
            if close[i] >= short_extreme + 0.20 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals