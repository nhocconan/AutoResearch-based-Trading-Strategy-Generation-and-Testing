#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_Trend_Regime
# Hypothesis: Use TRIX momentum (12-period) with volume spike confirmation and trend regime filter.
# TRIX > 0 indicates bullish momentum, TRIX < 0 bearish. Volume spike adds conviction.
# Trend regime: price above/below 50-period EMA on 4h timeframe.
# Designed for 20-50 trades/year per symbol, works in both bull and bear via trend filter.
# Uses 1d timeframe for additional regime filter (ADX) to avoid ranging markets.

name = "4h_TRIX_VolumeSpike_Trend_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate TRIX on 4h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1-period rate of change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change(1) * 100  # percentage change
    trix_values = trix.values

    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: current volume > 2.0x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    # ADX calculation on 1d for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - plus_dm_sum/period + plus_dm[i]
            minus_dm_sum = minus_dm_sum - minus_dm_sum/period + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period*2, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx

    # Calculate ADX on 1d
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # Align TRIX and EMA50 to 4h (already calculated on 4h, but ensure alignment)
    trix_aligned = trix_values  # Already on 4h timeframe
    ema_50_4h_aligned = ema_50_4h  # Already on 4h timeframe

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: only trade when ADX > 25 (trending market)
        trending_market = adx_1d_aligned[i] > 25

        # Trend filter from 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]

        if position == 0:
            # LONG: TRIX > 0 (bullish momentum) AND price above EMA50 AND volume spike AND trending market
            if trix_aligned[i] > 0 and price_above_ema and volume_ok[i] and trending_market:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX < 0 (bearish momentum) AND price below EMA50 AND volume spike AND trending market
            elif trix_aligned[i] < 0 and price_below_ema and volume_ok[i] and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative OR price falls below EMA50
            if trix_aligned[i] <= 0 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive OR price rises above EMA50
            if trix_aligned[i] >= 0 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals