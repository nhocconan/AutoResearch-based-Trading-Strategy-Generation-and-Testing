#!/usr/bin/env python3
"""
1d_1w_trix_volume_regime_v1
Strategy: 1d TRIX momentum with volume confirmation and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: TRIX (12-period) captures momentum shifts; long when TRIX > 0 and rising, short when TRIX < 0 and falling. Volume confirmation (>1.5x average) filters weak moves. Weekly EMA20 trend filter ensures alignment with higher timeframe trend. Designed for 30-100 trades over 4 years (7-25/year) with discrete sizing to minimize fee drag. Works in bull/bear via momentum + trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_trix_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # TRIX: triple EMA of price, then ROC
    # EMA1
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (EMA3_today - EMA3_yesterday) / EMA3_yesterday
    trix = np.zeros_like(close)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(12, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # TRIX momentum: rising = bullish, falling = bearish
        trix_rising = trix[i] > trix[i-1]
        trix_falling = trix[i] < trix[i-1]
        
        # Weekly trend filter
        uptrend_1w = price_close > ema_20_1w_aligned[i]
        downtrend_1w = price_close < ema_20_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: TRIX > 0 and rising with volume in uptrend
        long_signal = (trix[i] > 0) and trix_rising and vol_confirmed and uptrend_1w
        
        # Short: TRIX < 0 and falling with volume in downtrend
        short_signal = (trix[i] < 0) and trix_falling and vol_confirmed and downtrend_1w
        
        # Exit when TRIX crosses zero or loses momentum
        exit_long = position == 1 and (trix[i] <= 0 or not trix_rising)
        exit_short = position == -1 and (trix[i] >= 0 or not trix_falling)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals