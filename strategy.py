#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime_v2
Hypothesis: 4h timeframe with 1d Camarilla R3/S3 breakouts in direction of 1d EMA34 trend, confirmed by volume spike (>2x 20-bar MA) and 1d ADX > 25 regime filter. Uses ATR-based trailing stop to limit drawdown. Discrete position sizing (0.25) to minimize fee churn. Designed for lower frequency (target 20-50 trades/year) to avoid fee drag, works in bull/bear via trend alignment.
"""

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d bar's OHLC for Camarilla levels (R3/S3)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1d ADX regime filter (ADX > 25 for trending market)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0) if high_1d[i] - high_1d[i-1] > high_1d[i-1] - low_1d[i] else 0
        minus_dm[i] = max(high_1d[i-1] - low_1d[i], 0) if high_1d[i-1] - low_1d[i] > high_1d[i] - high_1d[i-1] else 0
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d_arr[i-1]), abs(low_1d[i] - close_1d_arr[i-1]))
    
    period = 14
    alpha = 1.0 / period
    atr_1d = np.zeros(len(df_1d))
    plus_dm_smooth = np.zeros(len(df_1d))
    minus_dm_smooth = np.zeros(len(df_1d))
    
    # Initial values
    if len(df_1d) >= period + 1:
        atr_1d[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, len(df_1d)):
        atr_1d[i] = atr_1d[i-1] * (1 - alpha) + alpha * tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - alpha) + alpha * plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - alpha) + alpha * minus_dm[i]
    
    plus_di_1d = np.zeros(len(df_1d))
    minus_di_1d = np.zeros(len(df_1d))
    dx_1d = np.zeros(len(df_1d))
    
    for i in range(period, len(df_1d)):
        if atr_1d[i] != 0:
            plus_di_1d[i] = 100 * plus_dm_smooth[i] / atr_1d[i]
            minus_di_1d[i] = 100 * minus_dm_smooth[i] / atr_1d[i]
        dx_1d[i] = 100 * abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i]) if (plus_di_1d[i] + minus_di_1d[i]) != 0 else 0
    
    adx_1d = np.zeros(len(df_1d))
    if len(df_1d) >= period*2 + 1:
        adx_1d[period*2] = np.mean(dx_1d[period+1:period*2+1]) if len(dx_1d[period+1:period*2+1]) > 0 else 0
    
    for i in range(period*2+1, len(df_1d)):
        adx_1d[i] = adx_1d[i-1] * (1 - alpha) + alpha * dx_1d[i]
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d ATR for trailing stop
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25  # Position size
    atr_multiplier = 3.0  # ATR trailing stop multiplier
    
    # Warmup: max of calculations
    start_idx = max(20, 1, 34, 28, period*2+1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending_regime = adx_val > 25
        
        # Entry conditions
        long_entry = (close_val > r3_val) and bullish_1d and vol_spike and trending_regime
        short_entry = (close_val < s3_val) and bearish_1d and vol_spike and trending_regime
        
        # Trailing stop conditions
        if position == 1:
            # Long position: stop if price drops below highest high since entry minus ATR*multiplier
            if i == start_idx or position == 0:  # New position or warmup
                highest_since_entry = close_val
            else:
                highest_since_entry = max(highest_since_entry, close_val)
            long_stop = highest_since_entry - atr_multiplier * atr_val
            stop_long = close_val < long_stop
        elif position == -1:
            # Short position: stop if price rises above lowest low since entry plus ATR*multiplier
            if i == start_idx or position == 0:  # New position or warmup
                lowest_since_entry = close_val
            else:
                lowest_since_entry = min(lowest_since_entry, close_val)
            short_stop = lowest_since_entry + atr_multiplier * atr_val
            stop_short = close_val > short_stop
        else:
            stop_long = False
            stop_short = False
            highest_since_entry = close_val
            lowest_since_entry = close_val
        
        # Exit conditions: price returns inside Camarilla levels or trend reversal or regime change or stoploss
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_entry = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            lowest_since_entry = close_val
        elif position == 1 and (close_val < r3_val or not bullish_1d or not trending_regime or stop_long):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s3_val or not bearish_1d or not trending_regime or stop_short):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime_v2"
timeframe = "4h"
leverage = 1.0