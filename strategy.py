#!/usr/bin/env python3
"""
Experiment #154: 1h RSI(14) mean reversion + 4h/1d trend filter + session filter
HYPOTHESIS: In choppy/ranging markets (2025+ test), RSI extremes combined with higher-timeframe trend filters provide high-probability mean-reversion entries. Using 4h for trend direction and 1d for regime filter reduces false signals. Session filter (08-20 UTC) avoids low-liquidity periods. Target: 60-150 total trades over 4 years (15-37/year) to balance statistical significance with fee drag on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_154_1h_rsi14_4h_1d_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === Pre-compute session filter ONCE (avoid datetime ops in loop) ===
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 1h Indicators: RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # === HTF: 4h data for trend direction (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = pd.Series(df_4h['close'].values)
    # EMA(21) for 4h trend
    ema_4h = close_4h.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    # Price vs EMA: 1 = uptrend, -1 = downtrend
    trend_4h = np.where(close > ema_4h_aligned, 1, -1)
    
    # === HTF: 1d data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    # ADX(14) for regime detection on 1d
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    # Regime: trending if ADX > 25, ranging if ADX <= 25
    ranging_regime = adx_1d_aligned <= 25
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # sufficient for RSI + HTF warmup
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any indicator is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Mean reversion completion or stoploss ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2.5*ATR(14) approximation using price deviation
            # Simplified: exit if price moves 2.5% against entry (conservative for 1h)
            if position_side > 0:  # Long
                if price < entry_price * 0.975:  # 2.5% stop
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > entry_price * 1.025:  # 2.5% stop
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit conditions: RSI returns to neutral zone (40-60)
            if 40 <= rsi[i] <= 60:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Time-based exit: max 12 bars (~12h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Only enter in ranging regime (ADX <= 25) for mean reversion
        if ranging_regime[i]:
            # Long: RSI oversold (<30) + 4h uptrend filter
            if rsi[i] < 30 and trend_4h[i] > 0:
                in_position = True
                position_side = 1
                entry_price = price
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: RSI overbought (>70) + 4h downtrend filter
            elif rsi[i] > 70 and trend_4h[i] < 0:
                in_position = True
                position_side = -1
                entry_price = price
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals