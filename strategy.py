#!/usr/bin/env python3
"""
Experiment #020: 12h Donchian Breakout + Volume + ATR Stop

HYPOTHESIS: Donchian(20) breakout is a proven price structure signal that works
in ALL markets (bull breaks up, bear breaks down). Combined with volume confirmation
and ATR-based stoploss, this captures institutional moves while limiting drawdown.

WHY BOTH BULL AND BEAR:
- Donchian breakout is symmetric: up breaks = longs, down breaks = shorts
- 2021 bull: ride the breakout upside
- 2022 bear: ride the breakout downside  
- Range: whipsaws but stoploss limits damage

TARGET: 75-150 total trades over 4 years (proven range for 12h).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (test Sharpe 1.382)

KEY DESIGN:
1. Donchian(20) breakout on 12h - simple price channel
2. Volume confirmation (>1.5x 20-avg)
3. 1d HMA for trend direction (bias entries with trend)
4. ATR(14) stoploss at 2x - tight but not too tight
5. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_atr_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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
    """Donchian Channel - upper = highest high, lower = lowest low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
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
    
    # Load 1d data for trend bias (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1d EMA for additional trend check
    ema_1d_21_raw = calculate_ema(df_1d['close'].values, 21)
    ema_1d_21_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_21_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for local trend
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup - need 20 bars for Donchian + ATR
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
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
        
        # === TREND BIAS (1d) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        price_above_1d_ema = close[i] > ema_1d_21_aligned[i] if not np.isnan(ema_1d_21_aligned[i]) else True
        bull_trend = price_above_1d_hma and price_above_1d_ema
        bear_trend = not price_above_1d_hma and not price_above_1d_ema
        
        # === LOCAL TREND (12h EMA) ===
        ema_bullish = ema_8[i] > ema_21[i] if (not np.isnan(ema_8[i]) and not np.isnan(ema_21[i])) else True
        ema_bearish = ema_8[i] < ema_21[i] if (not np.isnan(ema_8[i]) and not np.isnan(ema_21[i])) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN CHANNEL LEVELS ===
        upper = donch_upper[i]
        lower = donch_lower[i]
        middle = donch_mid[i]
        
        # Breakout detection: close above upper or below lower
        bullish_breakout = close[i] > upper
        bearish_breakout = close[i] < lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Bullish breakout + trend confirmation
        if bullish_breakout and not in_position:
            # Require trend alignment
            if bull_trend or (price_above_1d_hma and ema_bullish):
                # Volume confirmation strongly preferred
                if vol_spike:
                    desired_signal = SIZE
                elif ema_bullish:
                    desired_signal = SIZE * 0.5  # Half size without volume
        
        # SHORT: Bearish breakout + trend confirmation
        if bearish_breakout and not in_position:
            # Require trend alignment
            if bear_trend or (not price_above_1d_hma and ema_bearish):
                # Volume confirmation strongly preferred
                if vol_spike:
                    desired_signal = -SIZE
                elif ema_bearish:
                    desired_signal = -SIZE * 0.5  # Half size without volume
        
        # === STOPLOSS CHECK (trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === REVERSE ON STRONG OPPOSITE BREAKOUT ===
        if not in_position and desired_signal == 0.0:
            # Allow counter-trend entries only with very strong signal
            if bullish_breakout and vol_spike and not price_above_1d_hma:
                # Counter-trend, only with volume
                desired_signal = SIZE * 0.5
            if bearish_breakout and vol_spike and price_above_1d_hma:
                # Counter-trend, only with volume
                desired_signal = -SIZE * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals