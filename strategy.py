#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + 1w Trend Filter

HYPOTHESIS: Price channel breakouts identify institutional momentum shifts.
Weekly EMA(21) confirms the larger trend direction, filtering false breakouts.
Volume spike (>1.5x 20d avg) validates genuine moves.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Long entries only during bull markets when price breaks above 20d high
- Short entries only during bear markets when price breaks below 20d low
- ATR(14)-based stoploss adapts to volatility regime
- Weekly trend filter prevents trading against major trends

TARGET: 30-80 trades over 4 years (conservative entry = fewer trades)
Reference: mtf_1d_kama_rsi_chop_regime_1w_v1 (Sharpe=1.310, 74 trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_v1"
timeframe = "1d"
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

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend direction
    ema_1w_raw = calculate_ema(df_1w['close'].values, span=21)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_raw)
    
    # 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 days)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Weekly trend filter
        price_above_1w_ema = close[i] > ema_1w_aligned[i] if not np.isnan(ema_1w_aligned[i]) else True
        price_below_1w_ema = close[i] < ema_1w_aligned[i] if not np.isnan(ema_1w_aligned[i]) else False
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price breaks above 20d high + above weekly EMA + volume
        if close[i] > donchian_high[i] and price_above_1w_ema and vol_spike:
            desired_signal = SIZE
        
        # SHORT: Price breaks below 20d low + below weekly EMA + volume
        elif close[i] < donchian_low[i] and price_below_1w_ema and vol_spike:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals