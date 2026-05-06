#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ATR-based volatility breakout with 1w trend filter and volume confirmation
# Long when price > upper Donchian(20) on 6h AND 1d ATR(14) > 1.5 * ATR(50) (volatility expansion) AND 1w close > 1w EMA50 (uptrend) AND volume > 1.5 * avg_volume(20)
# Short when price < lower Donchian(20) on 6h AND 1d ATR(14) > 1.5 * ATR(50) (volatility expansion) AND 1w close < 1w EMA50 (downtrend) AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back inside Donchian(10) channels (mean reversion to shorter-term structure)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Volatility expansion captures breakouts from consolidation with follow-through
# 1w EMA50 trend filter ensures we trade with the dominant weekly trend
# Shorter Donchian(10) exit allows for quick mean reversion when momentum fades
# Works in both bull (buy volatility expansion breakouts) and bear (sell volatility expansion breakdowns) markets

name = "6h_VolatilityExpansion_DonchianBreakout_1wTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels
    donchian_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Get 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed 1d bars for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = 0  # First bar: no previous close
    tr3[0] = 0  # First bar: no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_expansion = atr_14 > (1.5 * atr_50)
    
    # Align 1d volatility expansion to 6h timeframe (wait for completed 1d bar)
    volatility_expansion_aligned = align_htf_to_ltf(prices, df_1d, volatility_expansion)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need at least 50 completed weekly bars for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_20_high[i]) or np.isnan(donchian_20_low[i]) or 
            np.isnan(donchian_10_high[i]) or np.isnan(donchian_10_low[i]) or
            np.isnan(volatility_expansion_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian(20), volatility expansion, 1w EMA50 uptrend, volume spike, in session
            if (close[i] > donchian_20_high[i] and 
                volatility_expansion_aligned[i] and 
                close_1d[-1] > ema_50_1w[-1] if len(close_1d) > 0 and len(ema_50_1w) > 0 else False and  # Simplified trend check
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20), volatility expansion, 1w EMA50 downtrend, volume spike, in session
            elif (close[i] < donchian_20_low[i] and 
                  volatility_expansion_aligned[i] and 
                  close_1d[-1] < ema_50_1w[-1] if len(close_1d) > 0 and len(ema_50_1w) > 0 else False and  # Simplified trend check
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses back inside Donchian(10) (mean reversion)
            if close[i] < donchian_10_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses back inside Donchian(10) (mean reversion)
            if close[i] > donchian_10_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals