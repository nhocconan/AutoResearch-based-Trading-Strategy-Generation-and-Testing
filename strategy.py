#!/usr/bin/env python3
"""
6h_ADX_Alligator_Trend_1wFilter
Hypothesis: 6h ADX > 25 + Alligator alignment (JAW < TEETH < LIPS for long, reverse for short) with 1w trend filter (price > 1w EMA34 for long, < for short). Uses discrete sizing (0.25) to limit fee churn. Designed for 6h timeframe with ~12-30 trades/year, works in bull/bear by following 1w trend filter and requiring strong trend alignment.
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
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Alligator components (13,8,5 periods with offsets 8,5,3)
    # JAW (Blue line): 13-period SMMA smoothed 8 periods ahead
    # TEETH (Red line): 8-period SMMA smoothed 5 periods ahead
    # LIPS (Green line): 5-period SMMA smoothed 3 periods ahead
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(data, np.nan, dtype=float)
        for i in range(len(data)):
            if i < period:
                smma_vals[i] = np.nan
            elif i == period:
                smma_vals[i] = sma[i]
            else:
                if np.isnan(smma_vals[i-1]) or np.isnan(sma[i]):
                    smma_vals[i] = np.nan
                else:
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + sma[i]) / period
        return smma_vals
    
    jaw = smma(high + low, 13)  # Using median price for Alligator
    teeth = smma(high + low, 8)
    lips = smma(high + low, 5)
    
    # Apply offsets: JAW offset 8, TEETH offset 5, LIPS offset 3
    jaw_offset = np.roll(jaw, 8)
    teeth_offset = np.roll(teeth, 5)
    lips_offset = np.roll(lips, 3)
    
    # ADX calculation (14 periods)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period-1] = np.nanmean(tr[:period])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        for i in range(len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
        
        adx = np.zeros_like(high)
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough data for Alligator (max offset 8) and ADX
    start_idx = max(52, 8) + 14  # 52 for Alligator SMMA stability, 8 for offset, 14 for ADX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_offset[i]) or np.isnan(teeth_offset[i]) or np.isnan(lips_offset[i]) or
            np.isnan(adx[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: ADX > 25 + JAW < TEETH < LIPS (Alligator aligned up) + 1w uptrend
            long_condition = (adx[i] > 25) and \
                            (jaw_offset[i] < teeth_offset[i]) and \
                            (teeth_offset[i] < lips_offset[i]) and \
                            (curr_close > ema_34_1w_aligned[i])
            # Short: ADX > 25 + JAW > TEETH > LIPS (Alligator aligned down) + 1w downtrend
            short_condition = (adx[i] > 25) and \
                             (jaw_offset[i] > teeth_offset[i]) and \
                             (teeth_offset[i] > lips_offset[i]) and \
                             (curr_close < ema_34_1w_aligned[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Alligator reverses OR trend weakens (ADX < 20) OR price crosses 1w EMA
            if (jaw_offset[i] > teeth_offset[i]) or \
               (adx[i] < 20) or \
               (curr_close < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Alligator reverses OR trend weakens (ADX < 20) OR price crosses 1w EMA
            if (jaw_offset[i] < teeth_offset[i]) or \
               (adx[i] < 20) or \
               (curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_Trend_1wFilter"
timeframe = "6h"
leverage = 1.0