#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX momentum with volume spike and weekly trend filter.
# TRIX (12-period) filters noise and captures momentum turns.
# Volume surge confirms breakout strength.
# Weekly EMA20 trend ensures alignment with higher timeframe trend.
# Designed for low-frequency, high-conviction trades in both bull and bear markets.
name = "12h_TRIX12_VolumeSpike_1wEMA20"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # TRIX (12-period) on 12h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.zeros(n)
    trix[12:] = (ema3[12:] - ema3[11:-1]) / ema3[11:-1] * 100  # percent change
    
    # Weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike: volume > 2.0x 30-period EMA
    vol_ema30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_spike = volume > (2.0 * vol_ema30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(trix[i]) or np.isnan(ema_20_12h[i]) or
            np.isnan(vol_ema30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: TRIX turns positive with volume spike and above weekly EMA20
            if (trix[i] > 0 and trix[i-1] <= 0 and vol_spike[i] and price > ema_20_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX turns negative with volume spike and below weekly EMA20
            elif (trix[i] < 0 and trix[i-1] >= 0 and vol_spike[i] and price < ema_20_12h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns negative (momentum reversal)
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns positive (momentum reversal)
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals