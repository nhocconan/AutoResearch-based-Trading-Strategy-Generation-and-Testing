#!/usr/bin/env python3
"""
Experiment #014: 1h Strategy with 4h/1d HTF Direction and Session Filter

HYPOTHESIS: Use 4h Donchian breakout direction and 1d EMA trend filter for signal direction,
enter on 1h pullbacks to EMA21 with RSI extremes and volume confirmation.
Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year to minimize fee drag.
Designed to work in both bull and bear markets via HTF trend alignment and mean-reversion entries.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_donchian_1d_ema_rsi_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper min_periods."""
    return pd.Series(values).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # === HTF: 4h Donchian(20) for direction (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    dc_upper_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().shift(1).values
    dc_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_upper_4h)
    dc_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_lower_4h)
    
    # === HTF: 1d EMA(50) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_50 = calculate_ema(df_1d['close'].values, 50)
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # === 1h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    ema_21 = calculate_ema(close, 21)
    rsi_14 = calculate_rsi(close, 14)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(ema_21[i]) or np.isnan(rsi_14[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(dc_upper_4h_aligned[i]) or 
            np.isnan(dc_lower_4h_aligned[i]) or np.isnan(ema_1d_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- HTF Direction ---
        # 4h Donchian breakout direction
        bullish_4h = close[i] > dc_upper_4h_aligned[i]
        bearish_4h = close[i] < dc_lower_4h_aligned[i]
        # 1d EMA trend
        trend_bullish = close[i] > ema_1d_50_aligned[i]
        trend_bearish = close[i] < ema_1d_50_aligned[i]
        
        # --- 1h Entry Conditions ---
        # Pullback to EMA21 with RSI extreme and volume confirmation
        near_ema = abs(close[i] - ema_21[i]) < (0.5 * atr_14[i])  # Within 0.5 ATR of EMA21
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based stoploss (2.0x)
            if position_side > 0:
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: HTF direction reversal or RSI normalization
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold
            if min_hold:
                if position_side > 0:
                    # Exit long: 4h turns bearish OR 1d trend turns bearish OR RSI > 50
                    if not bullish_4h or not trend_bullish or rsi_14[i] > 50:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: 4h turns bullish OR 1d trend turns bullish OR RSI < 50
                    if not bearish_4h or not trend_bearish or rsi_14[i] < 50:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # 4h bullish breakout AND 1d bullish trend AND pullback to EMA21 with RSI oversold AND volume
        if bullish_4h and trend_bullish and near_ema and rsi_oversold and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            signals[i] = SIZE
        # Short conditions:
        # 4h bearish breakout AND 1d bearish trend AND pullback to EMA21 with RSI overbought AND volume
        elif bearish_4h and trend_bearish and near_ema and rsi_overbought and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals