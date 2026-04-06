#!/usr/bin/env python3
"""
1h Volume Spike with 4h/1d Trend Filter
Hypothesis: Volume spikes on 1h confirm momentum, filtered by 4h trend and 1d regime.
Trades only in direction of 4h trend, avoids counter-trend whipsaws. 1d filter avoids
extreme volatility periods. Volume spike ensures institutional participation.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_spike_4h_1d_trend_v1"
timeframe = "1h"
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
    
    # 14-period ATR for stops and filters
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 50-period EMA on 4h for trend
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 49) / 51
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1d data for regime filter (avoid high volatility)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period ATR on 1d for volatility regime
    atr_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        tr_1d = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        if len(tr_1d) > 0:
            atr_1d[20] = np.mean(tr_1d[:20])
            for i in range(21, len(close_1d)):
                atr_1d[i] = (atr_1d[i-1] * 19 + tr_1d[i-1]) / 20
    
    # 50-period SMA on 1d for trend filter
    sma_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        for i in range(49, len(close_1d)):
            sma_1d[i] = np.mean(close_1d[i-49:i+1])
    
    # Volatility regime: 1 = low vol (good for trend), 0 = high vol (avoid)
    vol_regime = np.ones(len(close_1d))
    if len(close_1d) >= 50 and len(atr_1d) >= 50:
        atr_ratio = atr_1d / sma_1d  # ATR as % of price
        atr_ratio_ma = np.full(len(close_1d), np.nan)
        for i in range(49, len(close_1d)):
            atr_ratio_ma[i] = np.mean(atr_ratio[i-49:i+1])
        # Low volatility when ATR ratio below its 50-day average
        vol_regime = (atr_ratio < atr_ratio_ma).astype(float)
    
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Volume spike: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_4h_aligned[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(vol_regime_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume condition
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: 4h trend turns down OR volatility regime shifts to high vol
            # Stoploss: price drops 2.0*ATR below entry
            if (trend_4h_aligned[i] == -1 or
                vol_regime_aligned[i] == 0 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: 4h trend turns up OR volatility regime shifts to high vol
            # Stoploss: price rises 2.0*ATR above entry
            if (trend_4h_aligned[i] == 1 or
                vol_regime_aligned[i] == 0 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for volume spike entries in direction of 4h trend
            # Only trade in session and low volatility regime
            if in_session and vol_regime_aligned[i] == 1:
                # Long: volume spike in uptrend
                if volume_spike and trend_4h_aligned[i] == 1:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: volume spike in downtrend
                elif volume_spike and trend_4h_aligned[i] == -1:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals