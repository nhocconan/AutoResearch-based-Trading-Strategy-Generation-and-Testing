#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime filter
    # Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA
    # ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = ranging (fade extremes)
    # Works in bull/bear by adapting to regime: trend follow in trending, mean revert in ranging
    # Target: 12-37 trades/year per symbol (50-150 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 13-period EMA for Elder Ray
    def calculate_ema(values, period):
        ema = np.full(len(values), np.nan)
        if len(values) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema13 = calculate_ema(close_1d, 13)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Directional Movement
        dm_plus = np.zeros(len(high))
        dm_minus = np.zeros(len(high))
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0
        
        # Smoothed TR, DM+
        atr = np.zeros(len(high))
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth = np.zeros(len(high))
        dm_minus_smooth = np.zeros(len(high))
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros(len(high))
        di_minus = np.zeros(len(high))
        dx = np.zeros(len(high))
        for i in range(period, len(high)):
            if atr[i] != 0:
                di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
                di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
                if di_plus[i] + di_minus[i] != 0:
                    dx[i] = (abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
        
        # ADX (smoothed DX)
        adx = np.zeros(len(high))
        if len(dx) >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
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
        
        # Regime-based logic
        if adx_aligned[i] > 25:  # Trending market - follow Elder Ray
            # Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
            long_entry = bull_power_aligned[i] > 0 and (i == 100 or bull_power_aligned[i] > bull_power_aligned[i-1])
            short_entry = bear_power_aligned[i] < 0 and (i == 100 or bear_power_aligned[i] < bear_power_aligned[i-1])
            # Exit when power crosses zero
            long_exit = bull_power_aligned[i] <= 0
            short_exit = bear_power_aligned[i] >= 0
        else:  # Ranging market (ADX < 20) - fade extremes
            # Long when Bear Power < 0 (oversold), Short when Bull Power > 0 (overbought)
            long_entry = bear_power_aligned[i] < 0
            short_entry = bull_power_aligned[i] > 0
            # Exit when power returns to zero
            long_exit = bear_power_aligned[i] >= 0
            short_exit = bull_power_aligned[i] <= 0
        
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

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0