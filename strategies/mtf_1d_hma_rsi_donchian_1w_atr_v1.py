#!/usr/bin/env python3
"""
Experiment #1513: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback + Donchian Momentum

Hypothesis: Based on #1506 success (12h HMA+RSI), scaling to 1d with 1w HTF should work better.
Key insights from 1100+ failed strategies:
1. Complex regime filters (CHOP+CRSI) = 0 trades or negative Sharpe (#1501, #1507, #1511, #1512)
2. SIMPLER works: HTF trend bias + primary trend + RSI pullback (#1505, #1506 kept)
3. 1d timeframe naturally generates 20-50 trades/year (perfect for fee efficiency)
4. Donchian breakout adds momentum confirmation without over-filtering
5. Loose RSI bands (30-70) ensure trades happen while maintaining quality

Design:
- 1w HMA(21) for macro trend direction (HTF filter)
- 1d HMA(21) for primary trend confirmation
- 1d RSI(14) for pullback entries (loose: 30-70 range ensures trades)
- Donchian(20) breakout as momentum confirmation (price near 20d high/low)
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.30 (discrete: 0.0, ±0.30)
- Target: 30-60 trades/train (4 years), 8-15 trades/test (15 months)

Timeframe: 1d (as required by experiment)
HTF: 1w (weekly trend bias)
Position Size: 0.30 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_donchian_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(close) if 'close' in dir() else len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Donchian channels
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 1d (20-50 trades/year target)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - primary direction bias ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) - confirmation ===
        daily_bull = close[i] > hma_1d[i]
        daily_bear = close[i] < hma_1d[i]
        
        # === RSI PULLBACK - LOOSE bands for MORE trades ===
        # Long: RSI pulled back but not oversold (30-55)
        rsi_pullback_long = 30.0 <= rsi[i] <= 55.0
        # Short: RSI rallied but not overbought (45-70)
        rsi_pullback_short = 45.0 <= rsi[i] <= 70.0
        
        # === DONCHIAN MOMENTUM - price near channel bounds ===
        # Long: price in upper half of Donchian (momentum confirmation)
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 1e-10:
            donchian_position = (close[i] - donchian_lower[i]) / donchian_range
        else:
            donchian_position = 0.5
        
        donchian_bull = donchian_position > 0.5  # price in upper half
        donchian_bear = donchian_position < 0.5  # price in lower half
        
        # === DESIRED SIGNAL - SIMPLIFIED FOR 1d ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + RSI pullback + Donchian momentum
        # Option 1: Strong trend (1w + 1d both bull) + RSI pullback + Donchian bull
        if weekly_bull and daily_bull and rsi_pullback_long and donchian_bull:
            desired_signal = BASE_SIZE
        # Option 2: 1w bull + 1d bull + RSI pullback (looser, ensures trades)
        elif weekly_bull and daily_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE * 0.9
        # Option 3: 1w bull + 1d bull + Donchian bull (fallback for trades)
        elif weekly_bull and daily_bull and donchian_bull:
            desired_signal = BASE_SIZE * 0.7
        # Option 4: 1w bull + 1d above HMA + RSI not overbought (loosest)
        elif weekly_bull and daily_bull and rsi[i] < 65.0:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT: 1w bearish + 1d bearish + RSI pullback + Donchian momentum
        # Option 1: Strong trend (1w + 1d both bear) + RSI pullback + Donchian bear
        elif weekly_bear and daily_bear and rsi_pullback_short and donchian_bear:
            desired_signal = -BASE_SIZE
        # Option 2: 1w bear + 1d bear + RSI pullback (looser, ensures trades)
        elif weekly_bear and daily_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE * 0.9
        # Option 3: 1w bear + 1d bear + Donchian bear (fallback for trades)
        elif weekly_bear and daily_bear and donchian_bear:
            desired_signal = -BASE_SIZE * 0.7
        # Option 4: 1w bear + 1d below HMA + RSI not oversold (loosest)
        elif weekly_bear and daily_bear and rsi[i] > 35.0:
            desired_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals