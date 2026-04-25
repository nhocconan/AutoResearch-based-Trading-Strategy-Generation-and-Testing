#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Adaptive
Hypothesis: Use Elder Ray (Bull/Bear Power) with regime filter on 6h timeframe.
In bull regime (ADX>25 + price>SMA50): go long when Bull Power > 0 and rising.
In bear regime (ADX>25 + price<SMA50): go short when Bear Power < 0 and falling.
In range regime (ADX<20): mean revert at Bollinger Band extremes.
Uses 1d HTF for trend context and volume confirmation.
Target: 12-30 trades/year, size 0.25.
Works in bull via trend-following Elder Ray, in bear via mean reversion and short Elder Ray.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX on 6h for regime detection
    # +DI, -DI, DX calculation
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Calculate Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Bollinger Bands for mean reversion in ranging markets
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(sma20[i]) or np.isnan(std20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine regime
        is_bull_trend = adx[i] > 25 and close[i] > ema_50_1d_aligned[i]
        is_bear_trend = adx[i] > 25 and close[i] < ema_50_1d_aligned[i]
        is_range = adx[i] < 20
        
        if position == 0:
            # Entry logic based on regime
            if is_bull_trend:
                # Long when Bull Power positive and rising (momentum)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_bear_trend:
                # Short when Bear Power negative and falling (momentum)
                if bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif is_range:
                # Mean reversion at Bollinger extremes
                if close[i] <= lower_bb[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_bb[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            exit_long = False
            if is_bull_trend:
                # Exit when Bull Power turns negative
                if bull_power[i] <= 0:
                    exit_long = True
            elif is_bear_trend:
                # Exit when price crosses above 1d EMA50 (trend change)
                if close[i] > ema_50_1d_aligned[i]:
                    exit_long = True
            else:  # range
                # Exit when price returns to mean or opposite extreme
                if close[i] >= sma20[i] or close[i] >= upper_bb[i]:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            exit_short = False
            if is_bear_trend:
                # Exit when Bear Power turns positive
                if bear_power[i] >= 0:
                    exit_short = True
            elif is_bull_trend:
                # Exit when price crosses below 1d EMA50 (trend change)
                if close[i] < ema_50_1d_aligned[i]:
                    exit_short = True
            else:  # range
                # Exit when price returns to mean or opposite extreme
                if close[i] <= sma20[i] or close[i] <= lower_bb[i]:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0