#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA50 trend filter + volume spike
# Long when Williams %R crosses above -80 (oversold bounce) + 1d EMA50 uptrend + volume > 1.5x 20-period avg
# Short when Williams %R crosses below -20 (overbought rejection) + 1d EMA50 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 1d EMA50 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Williams %R(14) captures short-term reversals with proven edge in ranging markets.
# Volume threshold (1.5x) targets ~30-60 trades/year on 4h timeframe to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # === Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Where Highest High and Lowest Low are over the past 14 periods
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, 50, 20) + 5  # Williams %R(14) + EMA50 + volume(20) + buffer
    
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
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Williams %R thresholds
        wr_oversold = -80.0
        wr_overbought = -20.0
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (from below)
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (williams_r[i] > wr_oversold and williams_r[i-1] <= wr_oversold) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (from above)
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (williams_r[i] < wr_overbought and williams_r[i-1] >= wr_overbought) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR14_1dEMA50_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0