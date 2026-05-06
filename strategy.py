#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend + volume spike confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) on 12h for trend direction and entry timing
# 1d EMA34 filters for higher timeframe trend alignment to reduce whipsaw
# Volume spike (>2.0x 20-bar average) confirms breakout strength
# ATR-based trailing stop via signal=0 when price retraces 30% of ATR from extreme
# Designed for 12h timeframe to capture medium-term trends in both bull/bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 13 or len(df_1d) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 12h
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        sma = np.mean(source[:period])
        result[period-1] = sma
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_12h = smma(close_12h, 13)
    teeth_12h = smma(close_12h, 8)
    lips_12h = smma(close_12h, 5)
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price > EMA34 AND volume spike
            if (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.30
                position = 1
                long_extreme = close[i]
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price < EMA34 AND volume spike
            elif (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.30
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 30% of ATR from extreme
            if close[i] <= long_extreme - 0.3 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 30% of ATR from extreme
            if close[i] >= short_extreme + 0.3 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.30
    
    return signals