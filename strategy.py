#!/usr/bin/env python3
"""
1d_KAMA_Regime_Volume
Hypothesis: On daily timeframe, KAMA adapts to market noise and captures sustained trends.
In trending regimes (ADX>25), KAMA direction provides edge. In ranging regimes (ADX<20),
we fade moves at Bollinger Bands with volume confirmation. Weekly EMA50 filter ensures
we only trade with the higher timeframe trend. Designed for 30-100 trades over 4 years
on 1d timeframe, working in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast_len=2, slow_len=30):
    """Kaufman Adaptive Moving Average with min_periods"""
    if len(close) < er_len:
        return np.full_like(close, np.nan, dtype=float)
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s.diff(er_len))
    volatility = close_s.diff().abs().rolling(window=er_len, min_periods=er_len).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
    # KAMA
    kama = np.zeros_like(close, dtype=float)
    kama[0] = close_s.iloc[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    return kama

def calculate_adx(high, low, close, period=14):
    """ADX (Average Directional Index) with min_periods"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    # True Range
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    # Directional Movement
    up = high_s.diff()
    down = -low_s.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=high_s.index)
    minus_dm = pd.Series(minus_dm, index=high_s.index)
    # Smoothed DM
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)
    # DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx.fillna(0).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily KAMA
    kama = calculate_kama(close, er_len=10, fast_len=2, slow_len=30)
    
    # Daily ADX for regime detection
    adx = calculate_adx(high, low, close, period=14)
    
    # Daily Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = (bb_ma + 2 * bb_std).values
    bb_lower = (bb_ma - 2 * bb_std).values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = close_s.rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma * 1.5).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for KAMA (30), ADX (14), BB (20), volume MA (20)
    start_idx = max(30, 14, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma.iloc[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx[i] > 25:  # Trending regime
                # Trade in direction of KAMA slope and weekly EMA50
                kama_rising = kama[i] > kama[i-1]
                kama_falling = kama[i] < kama[i-1]
                weekly_uptrend = curr_close > ema_50_1w_aligned[i]
                weekly_downtrend = curr_close < ema_50_1w_aligned[i]
                
                if kama_rising and weekly_uptrend and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif kama_falling and weekly_downtrend and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging regime (ADX < 25)
                # Fade moves at Bollinger Bands with volume confirmation
                if curr_close <= bb_lower[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif curr_close >= bb_upper[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position exit conditions
            if adx[i] > 25:  # Trending regime
                # Exit when KAMA turns down or weekly trend breaks
                if kama[i] < kama[i-1] or curr_close < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when price reaches middle Bollinger Band
                bb_middle = (bb_upper[i] + bb_lower[i]) / 2
                if curr_close >= bb_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position exit conditions
            if adx[i] > 25:  # Trending regime
                # Exit when KAMA turns up or weekly trend breaks
                if kama[i] > kama[i-1] or curr_close > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when price reaches middle Bollinger Band
                bb_middle = (bb_upper[i] + bb_lower[i]) / 2
                if curr_close <= bb_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Volume"
timeframe = "1d"
leverage = 1.0