#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter with 12h trend confirmation
# Bull regime: ADX > 25 (trending) AND Bull Power > 0 (buying pressure) AND price > 12h EMA50 (uptrend)
# Bear regime: ADX > 25 (trending) AND Bear Power < 0 (selling pressure) AND price < 12h EMA50 (downtrend)
# Exit when ADX < 20 (range) or power signals reverse
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year on 6h.
# Works in bull (buy strength with trend) and bear (sell weakness with trend).

name = "6h_ElderRay_ADX_Regime_12hEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ADX (14) on 6h timeframe
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                          np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+, DM-
        tr_period = np.zeros_like(tr)
        dm_plus_period = np.zeros_like(dm_plus)
        dm_minus_period = np.zeros_like(dm_minus)
        
        # Initial values (simple average)
        tr_period[period-1] = np.mean(tr[:period])
        dm_plus_period[period-1] = np.mean(dm_plus[:period])
        dm_minus_period[period-1] = np.mean(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            tr_period[i] = tr_period[i-1] - (tr_period[i-1]/period) + tr[i]
            dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1]/period) + dm_plus[i]
            dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1]/period) + dm_minus[i]
        
        # Directional Indicators
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        
        # DX and ADX
        dx = np.zeros_like(di_plus)
        dx[period-1:] = 100 * np.abs(di_plus[period-1:] - di_minus[period-1:]) / (di_plus[period-1:] + di_minus[period-1:])
        
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period-1:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1]/period) + dx[i]
            
        return adx
    
    # Calculate ADX
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate Elder Ray: Bull Power and Bear Power
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for ADX and EMA calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(adx[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        
        # Regime and power conditions
        trending = adx[i] > 25
        ranging = adx[i] < 20
        bull_pressure = bull_power[i] > 0
        bear_pressure = bear_power[i] < 0
        uptrend_12h = curr_close > ema_50_12h_aligned[i]
        downtrend_12h = curr_close < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Trending AND bull pressure AND 12h uptrend
            if trending and bull_pressure and uptrend_12h:
                signals[i] = 0.25
                position = 1
            # Short: Trending AND bear pressure AND 12h downtrend
            elif trending and bear_pressure and downtrend_12h:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when: ranging OR bear pressure appears OR 12h trend turns down
            if ranging or bear_pressure or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when: ranging OR bull pressure appears OR 12h trend turns up
            if ranging or bull_pressure or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals