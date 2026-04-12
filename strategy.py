# 4h_1d_Camarilla_Breakout_Volume_Regime_v2
# Hypothesis: Tighten entry conditions by requiring price to close beyond H3/L3, volume > 2x average,
# and volatility regime filter (ATR < MA) to reduce trades to ~25-35/year. Uses mean-reversion exits
# at H4/L4. Works in bull/bear via 1d trend filter (price > EMA50 for long, < EMA50 for short).
# Target: <40 total trades over 4 years to minimize fee drag while maintaining edge.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_Regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's Camarilla levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day fallback
    
    range_1d = prev_high - prev_low
    
    h5 = prev_close + (range_1d * 1.1 / 2)
    h4 = prev_close + (range_1d * 1.1)
    h3 = prev_close + (range_1d * 1.1 / 4)
    l3 = prev_close - (range_1d * 1.1 / 4)
    l4 = prev_close - (range_1d * 1.1)
    l5 = prev_close - (range_1d * 1.1 / 2)
    
    # === DAILY EMA50 TREND FILTER ===
    ema50 = np.zeros_like(close_1d)
    ema50[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50[i] = 0.02 * close_1d[i] + 0.98 * ema50[i-1]  # alpha = 2/(50+1)
    
    # === DAILY VOLATILITY REGIME (ATR < MA) ===
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    atr_ma = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if i < 29:
            atr_ma[i] = np.nan
        else:
            atr_ma[i] = np.mean(atr[i-29:i+1])
    vol_regime = atr < atr_ma  # True when low volatility (trending)
    
    # Align all daily data to 4h
    h3_a = align_htf_to_ltf(prices, df_1d, h3)
    l3_a = align_htf_to_ltf(prices, df_1d, l3)
    h4_a = align_htf_to_ltf(prices, df_1d, h4)
    l4_a = align_htf_to_ltf(prices, df_1d, l4)
    ema50_a = align_htf_to_ltf(prices, df_1d, ema50)
    vol_regime_a = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    # Volume average (20-period for confirmation)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_a[i]) or np.isnan(l3_a[i]) or np.isnan(h4_a[i]) or 
            np.isnan(l4_a[i]) or np.isnan(ema50_a[i]) or np.isnan(vol_regime_a[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TIGHTENED CONDITIONS
        vol_confirm = volume[i] > 2.0 * vol_avg[i]  # Increased from 1.5x to 2.0x
        in_trend_regime = vol_regime_a[i] > 0.5
        
        # LONG: price closes above H3, volume spike, low vol regime, price > EMA50 (uptrend)
        long_setup = (close[i] > h3_a[i]) and vol_confirm and in_trend_regime and (close[i] > ema50_a[i])
        # SHORT: price closes below L3, volume spike, low vol regime, price < EMA50 (downtrend)
        short_setup = (close[i] < l3_a[i]) and vol_confirm and in_trend_regime and (close[i] < ema50_a[i])
        
        # EXIT: mean reversion to opposite H4/L4
        exit_long = close[i] < l4_a[i]
        exit_short = close[i] > h4_a[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals