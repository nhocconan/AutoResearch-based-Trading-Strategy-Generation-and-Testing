#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend + volume spike
# Long when Williams %R < -80 (oversold) + 1d EMA34 uptrend + volume > 2.0x 24-period avg
# Short when Williams %R > -20 (overbought) + 1d EMA34 downtrend + volume > 2.0x 24-period avg
# Williams %R identifies exhaustion points in both bull and bear markets.
# 1d EMA34 provides strong trend filter reducing whipsaws.
# Volume threshold (2.0x) targets ~15-35 trades/year on 6h timeframe to avoid overtrading.
# Uses discrete position sizing (0.25) to control drawdown and minimize fee churn.

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high_14 - lowest_low_14
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    williams_r = ((highest_high_14 - close) / hl_range) * -100
    
    # Volume SMA for confirmation (using 24-period)
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 24, 14) + 5  # EMA34 + Williams(14) + volume(24) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. 1d EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (williams_r[i] < -80.0) and \
           (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. 1d EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (williams_r[i] > -20.0) and \
             (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0