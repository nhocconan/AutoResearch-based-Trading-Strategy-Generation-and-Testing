#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# In strong trends (ADX > 25): trade with Elder Ray direction (long if Bull Power > 0 and rising, short if Bear Power > 0 and rising)
# In weak trends/ranging (ADX <= 25): fade Elder Ray extremes (long if Bear Power < 0 and falling, short if Bull Power < 0 and falling)
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: trend following captures moves, regime filter avoids whipsaws in ranging markets

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
    
    # Calculate 1d EMA(13) for Elder Ray
    close_s_1d = pd.Series(close_1d)
    ema13_1d = close_s_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA(13)
    bear_power_1d = ema13_1d - low_1d   # Bear Power = EMA(13) - Low
    
    # Calculate 1d ADX(14) for trend strength
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing for TR and DM
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align 1d indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: strong trend vs weak trend/ranging
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] <= 25
        
        if position == 1:  # Long position
            if strong_trend:
                # Exit long if Bull Power turns negative or starts falling
                if bull_power_1d_aligned[i] <= 0 or (i > 50 and bull_power_1d_aligned[i] < bull_power_1d_aligned[i-1]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # weak_trend
                # Exit long if Bear Power rises above zero (fade failure)
                if bear_power_1d_aligned[i] > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if strong_trend:
                # Exit short if Bear Power turns negative or starts falling
                if bear_power_1d_aligned[i] <= 0 or (i > 50 and bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # weak_trend
                # Exit short if Bull Power rises above zero (fade failure)
                if bull_power_1d_aligned[i] > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if strong_trend:
                # Enter long on rising positive Bull Power
                if bull_power_1d_aligned[i] > 0 and (i <= 50 or bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1]):
                    position = 1
                    signals[i] = 0.25
                # Enter short on rising positive Bear Power
                elif bear_power_1d_aligned[i] > 0 and (i <= 50 or bear_power_1d_aligned[i] > bear_power_1d_aligned[i-1]):
                    position = -1
                    signals[i] = -0.25
            else:  # weak_trend
                # Fade extremes: long on falling negative Bear Power, short on falling negative Bull Power
                if bear_power_1d_aligned[i] < 0 and (i > 50 and bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]):
                    position = 1
                    signals[i] = 0.25
                elif bull_power_1d_aligned[i] < 0 and (i > 50 and bull_power_1d_aligned[i] < bull_power_1d_aligned[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals