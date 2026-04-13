#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + ADX regime filter with 1w trend
    # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND price > 1w EMA50
    # Short: Bull Power < 0 AND Bear Power > 0 AND ADX > 25 AND price < 1w EMA50
    # Exit: ADX < 20 (regime change to ranging) or power signals reverse
    # Using 1w for major trend, 6h for entry/exit timing
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time'].values
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for major trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Elder Ray and ADX on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ADX calculation (14-period)
    # +DM = max(High - High_prev, 0) if High - High_prev > Low_prev - Low else 0
    # -DM = max(Low_prev - Low, 0) if Low_prev - Low > High - High_prev else 0
    # TR = max(High - Low, High - Close_prev, Low - Close_prev)
    # +DI = 100 * EMA(+DM, 14) / ATR(14)
    # -DI = 100 * EMA(-DM, 14) / ATR(14)
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    # ADX = EMA(DX, 14)
    
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    plus_dm = np.where((high - high_prev) > (low_prev - low), np.maximum(high - high_prev, 0), 0)
    minus_dm = np.where((low_prev - low) > (high - high_prev), np.maximum(low_prev - low, 0), 0)
    
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (EMA with alpha=1/period)
    def wilders_ema(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr14 = wilders_ema(tr, 14)
    plus_di14 = 100 * wilders_ema(plus_dm, 14) / atr14
    minus_di14 = 100 * wilders_ema(minus_dm, 14) / atr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10)
    adx = wilders_ema(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA50
        long_trend_ok = close[i] > ema_1w_aligned[i]
        short_trend_ok = close[i] < ema_1w_aligned[i]
        
        # ADX regime filter: only trade in trending markets (ADX > 25)
        strong_trend = adx[i] > 25
        ranging_market = adx[i] < 20  # Exit condition
        
        # Elder Ray signals
        long_signal = bull_power[i] > 0 and bear_power[i] < 0
        short_signal = bull_power[i] < 0 and bear_power[i] > 0
        
        # Entry logic: Elder Ray + ADX + 1w trend
        long_entry = long_signal and strong_trend and long_trend_ok
        short_entry = short_signal and strong_trend and short_trend_ok
        
        # Exit logic: ADX < 20 (ranging) or signal reversal
        long_exit = ranging_market or (bull_power[i] <= 0) or (bear_power[i] >= 0)
        short_exit = ranging_market or (bull_power[i] >= 0) or (bear_power[i] <= 0)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_elder_ray_adx_trend_v1"
timeframe = "6h"
leverage = 1.0