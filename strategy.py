#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1w EMA34 trend filter and volume spike
# Long when Williams %R(14) crosses above -80 (oversold bounce) + 1w EMA34 uptrend + volume > 2.0x 24-period avg
# Short when Williams %R(14) crosses below -20 (overbought rejection) + 1w EMA34 downtrend + volume > 2.0x 24-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1w EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~15-25 trades/year to minimize fee drag on 12h timeframe.
# Williams %R calculated from 12h OHLC using 14-period lookback.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w Indicator: EMA34 ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 12h Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Where Highest High = max(high over lookback), Lowest Low = min(low over lookback)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if not (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or highest_high[i] == lowest_low[i]):
            williams_r[i] = ((highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])) * -100
    
    # Volume SMA for confirmation (using 24-period = 12 days of 12h bars)
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, lookback, 24) + 5  # EMA34 + Williams %R + volume(24) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # Williams %R levels
        williams_prev = williams_r[i-1] if i > 0 else williams_r[i]
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (from below -80 to above -80)
        # 2. 1w EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        williams_cross_up = (williams_prev <= -80.0) and (williams_r[i] > -80.0)
        if williams_cross_up and \
           (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (from above -20 to below -20)
        # 2. 1w EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (williams_prev >= -20.0) and (williams_r[i] < -20.0) and \
             (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR_1wEMA34_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0