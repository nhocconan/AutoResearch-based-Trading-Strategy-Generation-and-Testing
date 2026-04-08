#!/usr/bin/env python3
# 1d_bollinger_breakout_volume_regime_v1
# Hypothesis: Bollinger Band breakout with volume confirmation and regime filter (ADX) on daily timeframe.
# Captures volatility expansion in both bull and bear markets by trading breakouts during trending conditions.
# Uses BB(20,2) for breakout signals, volume >1.5x average for confirmation, ADX(14)>25 for trend filter.
# Target: 15-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_bollinger_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (weekly EMA50) - load once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily indicators
    # Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2.0 * std20
    lower_band = sma20 - 2.0 * std20
    
    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(sma20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(adx[i]) or np.isnan(avg_volume[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to middle band or trend weakens
            if close[i] < sma20[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle band or trend weakens
            if close[i] > sma20[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Breakout conditions with trend and weekly filter
            if volume_ok and adx[i] > 25:
                # Long breakout: price closes above upper band in uptrend
                if weekly_uptrend and close[i] > upper_band[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below lower band in downtrend
                elif weekly_downtrend and close[i] < lower_band[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals