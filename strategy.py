#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_TRIX_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX on 6h: 15-period EMA of price, then 15-period EMA of that, then 15-period EMA of that, then % change
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - previous value, then divided by previous EMA(EMA(EMA(close,15),15),15)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ema3.pct_change() * 100  # percentage change
    trix = trix_raw.fillna(0).values
    
    # Volume spike: current volume > 2.0 * 20-period SMA of volume
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_sma20)
    
    # Weekly trend filter: EMA34 on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for TRIX and volume calculations
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(trix[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero (bullish momentum) + weekly uptrend + volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and  # TRIX bullish crossover
                close[i] > ema34_1w_aligned[i] and    # weekly uptrend
                volume_spike[i]):                     # volume confirmation
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: TRIX crosses below zero (bearish momentum) + weekly downtrend + volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and  # TRIX bearish crossover
                  close[i] < ema34_1w_aligned[i] and    # weekly downtrend
                  volume_spike[i]):                     # volume confirmation
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: TRIX turns negative or weekly trend fails
            if (trix[i] < 0 or                    # TRIX bearish
                close[i] < ema34_1w_aligned[i]):  # weekly trend fail
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns positive or weekly trend fails
            if (trix[i] > 0 or                    # TRIX bullish
                close[i] > ema34_1w_aligned[i]):  # weekly trend fail
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals