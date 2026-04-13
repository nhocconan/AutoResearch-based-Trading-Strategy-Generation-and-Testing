#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + ADX regime filter + volume confirmation
    # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume > 1.5x 20-period average
    # Short: Bull Power < 0 AND Bear Power > 0 AND ADX > 25 AND volume > 1.5x 20-period average
    # Exit: ADX < 20 (regime change to ranging) or opposing Elder Ray signal
    # Uses 1d EMA13 for Elder Ray power calculations (trend reference)
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 12-37 trades/year (~50-150 over 4 years) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 (Elder Ray trend reference) - call ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA13 for Elder Ray calculations
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 6h ADX calculation (using Wilder's smoothing)
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
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_di[period] = 100 * np.mean(plus_dm[1:period+1]) / atr[period]
        minus_di[period] = 100 * np.mean(minus_dm[1:period+1]) / atr[period]
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = 100 * (plus_di[i-1] * (period-1) + plus_dm[i]) / (atr[i] * period)
            minus_di[i] = 100 * (minus_di[i-1] * (period-1) + minus_dm[i]) / (atr[i] * period)
        
        dx = np.zeros_like(tr)
        adx = np.zeros_like(tr)
        for i in range(period+1, len(tr)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
            if i == period+1:
                adx[i] = np.mean(dx[period+1:i+1])
            else:
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # ADX regime filter: only trade when trending (ADX > 25)
        adx_filter = adx[i] > 25
        
        # Elder Ray signals
        long_signal = (bull_power[i] > 0) and (bear_power[i] < 0)
        short_signal = (bull_power[i] < 0) and (bear_power[i] > 0)
        
        # Entry logic: Elder Ray + ADX + volume
        long_entry = long_signal and adx_filter and vol_confirm
        short_entry = short_signal and adx_filter and vol_confirm
        
        # Exit logic: ADX < 20 (ranging) or opposing Elder Ray signal
        long_exit = (adx[i] < 20) or short_signal
        short_exit = (adx[i] < 20) or long_signal
        
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

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0