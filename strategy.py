#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 12h ADX regime filter
    # Elder Ray (Bull/Bear power) measures trend strength via EMA13
    # ADX > 25 confirms trending regime for continuation trades
    # Works in bull/bear by only taking trades in strong trends
    # Target: 12-30 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Elder Ray and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA13 for Elder Ray
    def ema(values, span):
        result = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) < span:
            return result
        multiplier = 2 / (span + 1)
        result[span-1] = np.mean(values[:span])
        for i in range(span, len(values)):
            result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema13_12h = ema(close_12h, 13)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.full(len(high), np.nan)
        for i in range(len(high)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.full(len(high), np.nan)
        minus_dm = np.full(len(high), np.nan)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Smoothed TR, Plus_DM, Minus_DM (using Wilder's smoothing)
        def wilder_smooth(values, period):
            result = np.full_like(values, np.nan)
            if len(values) < period:
                return result
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = np.full(len(high), np.nan)
        minus_di = np.full(len(high), np.nan)
        for i in range(len(high)):
            if not np.isnan(tr_smooth[i]) and tr_smooth[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100
        
        # DX and ADX
        dx = np.full(len(high), np.nan)
        for i in range(len(high)):
            if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                di_sum = plus_di[i] + minus_di[i]
                if di_sum != 0:
                    dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100
        
        # ADX is smoothed DX
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align 12h indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (strong trend)
        if adx_aligned[i] <= 25:
            # No trend - stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Elder Ray logic in trending regime
        # Long when Bull Power > 0 and rising
        # Short when Bear Power < 0 and falling
        long_entry = bull_power_aligned[i] > 0 and (i == 100 or bull_power_aligned[i] > bull_power_aligned[i-1])
        short_entry = bear_power_aligned[i] < 0 and (i == 100 or bear_power_aligned[i] < bear_power_aligned[i-1])
        long_exit = bull_power_aligned[i] <= 0
        short_exit = bear_power_aligned[i] >= 0
        
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

name = "6h_12h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0