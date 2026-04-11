#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA for TRIX calculation
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # TRIX: percentage change in triple EMA
    trix_raw = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix = np.concatenate([[np.nan], trix_raw])
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_4h = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # 4x ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4x volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4x ADX for trend strength (14 period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_dm = tr[1:]
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_4h[i]) or np.isnan(trix_signal_4h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.3x average)
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Trend filter: ADX > 20 (moderate trend to allow more opportunities)
        trend_filter = adx[i] > 20
        
        # TRIX crossover signals
        trix_cross_up = trix_4h[i] > trix_signal_4h[i] and trix_4h[i-1] <= trix_signal_4h[i-1]
        trix_cross_down = trix_4h[i] < trix_signal_4h[i] and trix_4h[i-1] >= trix_signal_4h[i-1]
        
        # Long conditions: TRIX crosses up with volume and trend
        long_signal = volume_confirmed and trend_filter and trix_cross_up
        
        # Short conditions: TRIX crosses down with volume and trend
        short_signal = volume_confirmed and trend_filter and trix_cross_down
        
        # Exit when TRIX crosses in opposite direction
        exit_long = position == 1 and trix_cross_down
        exit_short = position == -1 and trix_cross_up
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily TRIX crossover strategy for 4h timeframe with volume confirmation (>1.3x average volume) and ADX filter (>20).
# Enters long when daily TRIX crosses above its signal line with volume >1.3x average and ADX>20.
# Enters short when daily TRIX crosses below its signal line with same conditions.
# Exits when TRIX crosses in the opposite direction.
# Uses moderate ADX threshold to balance trade frequency and trend strength.
# Position size 0.25 to manage risk while capturing medium-term momentum.
# Target: 30-60 trades per year to minimize fee drag while maintaining edge.
# TRIX is effective in both bull and bear markets as it captures momentum shifts.
# Works on BTC/ETH as it adapts to changing market conditions through EMA smoothing.