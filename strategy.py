# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d Trend Filter and Volume Confirmation
# Uses Elder Ray (EMA13-based bull/bear power) to measure bull/bear strength.
# Filters trades by 1d EMA50 trend direction to align with higher timeframe bias.
# Requires volume > 1.5x 20-period average for institutional confirmation.
# Aims to capture trend continuations in both bull and bear markets with controlled risk.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Position size: 0.25 (25% of capital) to balance return and drawdown.

name = "6h_1d_elder_ray_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray components (requires EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull power: high - EMA13
    bear_power = low - ema13   # Bear power: low - EMA13
    
    # Align daily EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        ema13_val = ema13[i]
        bull = bull_power[i]
        bear = bear_power[i]
        ema50 = ema50_1d_aligned[i]
        volume_current = volume[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = price_close > ema50
        downtrend = price_close < ema50
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: bull power positive AND uptrend AND volume confirmation
        if bull > 0 and uptrend and volume_confirmed:
            long_signal = True
        
        # Short: bear power negative AND downtrend AND volume confirmation
        if bear < 0 and downtrend and volume_confirmed:
            short_signal = True
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and bull <= 0:  # Exit long when bull power fades
            position = 0
            signals[i] = 0.0
        elif position == -1 and bear >= 0:  # Exit short when bear power fades
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Elder Ray (Bull/Bear Power) measures the power of bulls/bears relative to EMA13.
# Bull Power = High - EMA13 (strength of bulls)
# Bear Power = Low - EMA13 (weakness of bears)
# Trades taken in direction of higher timeframe trend (daily EMA50) with volume confirmation.
# Exits when the respective power diminishes, avoiding hard stops that may cause whipsaws.
# Works in bull markets (buy bull power dips in uptrend) and bear markets (sell bear power rallies in downtrend).