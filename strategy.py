#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily Donchian(20) breakout levels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA20 weekly, ATR, Donchian, volume MA
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema20_1w_aligned[i]
        upper_donch = donch_high[i]
        lower_donch = donch_low[i]
        vol_spike_val = vol_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price closes above upper Donchian + volume spike + weekly uptrend (price > weekly EMA20)
            if close[i] > upper_donch and vol_spike_val and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: price closes below lower Donchian + volume spike + weekly downtrend (price < weekly EMA20)
            elif close[i] < lower_donch and vol_spike_val and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below weekly EMA20 or ATR-based trailing stop
            if close[i] < weekly_trend or close[i] < (high[i] - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above weekly EMA20 or ATR-based trailing stop
            if close[i] > weekly_trend or close[i] > (low[i] + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_Breakout_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0