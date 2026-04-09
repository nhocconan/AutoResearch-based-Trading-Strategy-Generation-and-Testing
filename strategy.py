#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter
# Elder Ray measures bull/bear power relative to EMA13: 
#   Bull Power = High - EMA13, Bear Power = Low - EMA13
# In strong trends (ADX > 25): follow Elder Ray signals
# In weak trends/ranging (ADX <= 25): fade extreme Elder Ray readings
# Uses 1d EMA13 for HTF alignment and 1d ADX for regime
# Discrete sizing 0.25 targets ~12-37 trades/year to minimize fee drag
# Works in bull/bear: trend following in strong moves, mean reversion in chop

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13
    close_s_1d = pd.Series(close_1d)
    ema13_1d = close_s_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d True Range for ADX
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate 1d +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing for TR, +DM, -DM
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Calculate 1d ADX
    plus_di = np.where(atr_1d > 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di = np.where(atr_1d > 0, 100 * minus_dm_smooth / atr_1d, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: strong trend vs weak trend/ranging
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] <= 25
        
        if position == 1:  # Long position
            if strong_trend:
                # Exit long if bull power turns negative (momentum fading)
                if bull_power[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # weak_trend
                # Exit long if bear power becomes extremely negative (oversold bounce fading)
                if bear_power[i] < -np.std(bull_power[max(0, i-50):i+1]) * 2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if strong_trend:
                # Exit short if bear power turns positive (selling pressure fading)
                if bear_power[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # weak_trend
                # Exit short if bull power becomes extremely positive (overbought fade fading)
                if bull_power[i] > np.std(bear_power[max(0, i-50):i+1]) * 2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if strong_trend:
                # Follow Elder Ray in strong trends
                if bull_power[i] > 0 and bear_power[i] < 0:
                    position = 1
                    signals[i] = 0.25
                elif bear_power[i] > 0 and bull_power[i] < 0:
                    position = -1
                    signals[i] = -0.25
            else:  # weak_trend
                # Fade extreme Elder Ray readings in ranging markets
                bull_power_ma = np.mean(bull_power[max(0, i-20):i+1])
                bear_power_ma = np.mean(bear_power[max(0, i-20):i+1])
                
                if bull_power[i] < bull_power_ma - np.std(bull_power[max(0, i-20):i+1]) * 1.5:
                    position = 1
                    signals[i] = 0.25
                elif bear_power[i] < bear_power_ma - np.std(bear_power[max(0, i-20):i+1]) * 1.5:
                    position = -1
                    signals[i] = -0.25
    
    return signals