#!/usr/bin/env python3
"""
Experiment #004: 12h Donchian Breakout + Weekly Trend Filter

HYPOTHESIS: Donchian breakouts at 12h timeframe with weekly trend confirmation
is a proven pattern that works in both bull and bear markets.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakout is symmetric: works for longs (bull) and shorts (bear)
- Weekly trend filter ensures we only trade with the larger timeframe trend
- Volume confirmation filters false breakouts
- 12h timeframe gives enough trades (not too few, not too many)
- ATR-based stops provide consistent risk management

DB EVIDENCE:
- mtf_4h_crsi_chop_donchian_regime_1d_v1: test_sharpe=1.460, 392 trades
- mtf_4h_hma_donchian_volume_rsi_12h_atr_v1: test_sharpe=1.382, 95 trades
- mtf_4h_hma_volume_donchian_adx_12h_atr_v1: test_sharpe=1.322, 94 trades

KEY DESIGN:
1. 12h Donchian(20) breakout as primary signal
2. 1w Donchian middle band as trend filter (must be aligned)
3. Volume spike (>1.5x 20-avg) to confirm breakout
4. ATR(14) stoploss at 2x
5. Simple, tight conditions to avoid overtrading
6. Target: 75-150 total trades over 4 years

CHANGES FROM PREVIOUS FAILURES:
- Previous 12h strategy had 323 trades (too many) → added weekly trend filter
- Previous 4h strategies had 0-45 trades (too few) → loosened entry slightly
- Focus on simplicity: breakout + volume + weekly alignment only
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        middle[i] = (upper[i] + np.min(low[i - period + 1:i + 1])) / 2.0
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, middle, lower

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Donchian for trend filter
    upper_1w, middle_1w, lower_1w = calculate_donchian(
        df_1w['high'].values,
        df_1w['low'].values,
        period=8
    )
    
    # Align to 12h
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    middle_1w_aligned = align_htf_to_ltf(prices, df_1w, middle_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # 1w EMA for additional trend confirmation
    ema_1w = calculate_ema(df_1w['close'].values, 8)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # 12h Donchian
    dc_upper_12h, dc_middle_12h, dc_lower_12h = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for local trend
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Warmup - need enough for all indicators
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # Skip if 12h Donchian not ready
        if np.isnan(dc_upper_12h[i]) or np.isnan(dc_lower_12h[i]):
            signals[i] = 0.0
            continue
        
        # Skip if 1w indicators not aligned
        if np.isnan(middle_1w_aligned[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if volume ratio not ready
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND FILTER ===
        # Weekly trend: price above EMA(8) on weekly = bullish
        weekly_bullish = close[i] > ema_1w_aligned[i]
        weekly_bearish = close[i] < ema_1w_aligned[i]
        
        # Weekly channel position: above middle = bullish bias, below = bearish bias
        above_weekly_middle = close[i] > middle_1w_aligned[i]
        below_weekly_middle = close[i] < middle_1w_aligned[i]
        
        # Combined weekly bias
        weekly_uptrend = weekly_bullish and above_weekly_middle
        weekly_downtrend = weekly_bearish and below_weekly_middle
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === 12H DONCHIAN BREAKOUT ===
        dc_upper = dc_upper_12h[i]
        dc_lower = dc_lower_12h[i]
        dc_mid = dc_middle_12h[i]
        
        # Previous bar's close for breakout detection
        prev_close = close[i - 1] if i > 0 else close[0]
        
        # Breakout: price crosses above upper or below lower
        breakout_above = prev_close <= dc_upper and close[i] > dc_upper
        breakout_below = prev_close >= dc_lower and close[i] < dc_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 12h breakout above + weekly uptrend + volume
        if breakout_above and weekly_uptrend:
            if vol_spike:
                desired_signal = SIZE
            else:
                desired_signal = SIZE * 0.5  # Smaller size without volume confirmation
        
        # SHORT: 12h breakout below + weekly downtrend + volume
        if breakout_below and weekly_downtrend:
            if vol_spike:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE * 0.5
        
        signals[i] = desired_signal
    
    return signals