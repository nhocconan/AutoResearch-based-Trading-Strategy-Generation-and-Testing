#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) mean reversion with 1w EMA200 trend filter and volume spike
# Long when Williams %R < -80 (oversold) + price > 1w EMA200 (uptrend) + volume > 2.0x 20-period avg
# Short when Williams %R > -20 (overbought) + price < 1w EMA200 (downtrend) + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Williams %R identifies exhaustion points in both bull and bear markets.
# 1w EMA200 provides strong trend filter reducing whipsaws and ensuring trades with major trend.
# Volume threshold targets ~15-30 trades/year on 6h timeframe to avoid overtrading.

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
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # === 1w Indicator: EMA200 ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(200, 14, 20) + 5  # EMA200 + Williams %R(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. Price > 1w EMA200 (uptrend)
        # 3. Volume confirmation
        if (williams_r[i] < -80) and \
           (close[i] > ema_200_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. Price < 1w EMA200 (downtrend)
        # 3. Volume confirmation
        elif (williams_r[i] > -20) and \
             (close[i] < ema_200_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_1wEMA200_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0