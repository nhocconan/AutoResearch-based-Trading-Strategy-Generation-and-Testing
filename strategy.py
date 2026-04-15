#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume spike
# Long when Williams %R crosses above -80 from below + 1d EMA50 uptrend + volume > 2.0x 20-period avg
# Short when Williams %R crosses below -20 from above + 1d EMA50 downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Williams %R (14) identifies overextended reversals; 1d EMA50 provides strong trend filter reducing whipsaws.
# Volume threshold (2.0x) targets ~20-35 trades/year on 4h timeframe to avoid overtrading.
# Effective in both bull (buying dips) and bear (selling rallies) markets via mean reversion within trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams %R (14) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 14, 20) + 5  # EMA50 + Williams %R(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Williams %R crossover conditions (using prior bar to avoid look-ahead)
        wr_prev = williams_r[i-1]
        wr_curr = williams_r[i]
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 from below (wr_prev <= -80 and wr_curr > -80)
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (wr_prev <= -80.0) and (wr_curr > -80.0) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 from above (wr_prev >= -20 and wr_curr < -20)
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (wr_prev >= -20.0) and (wr_curr < -20.0) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR_1dEMA50_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0