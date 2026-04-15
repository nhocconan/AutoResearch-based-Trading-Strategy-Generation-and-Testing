#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R(14) mean reversion with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) + 1d EMA50 uptrend + volume > 1.8x 20-period avg
# Short when Williams %R crosses below -20 (overbought) + 1d EMA50 downtrend + volume > 1.8x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Williams %R captures short-term reversals in both bull and bear markets.
# 1d EMA50 provides strong trend filter reducing whipsaws. Volume confirmation ensures participation.
# Target: 20-40 trades/year on 4h timeframe to avoid overtrading.

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
    
    # === Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: 0 to -100, where > -20 is overbought, < -80 is oversold
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100
    
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
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (bullish reversal from oversold)
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (williams_r[i] > -80) and (williams_r[i-1] <= -80) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (bearish reversal from overbought)
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (williams_r[i] < -20) and (williams_r[i-1] >= -20) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR14_1dEMA50_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0