#!/usr/bin/env python3
"""
Experiment #1509: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Based on experiment history, 4h timeframe with Donchian+HMA+RSI pattern works:
- #1499 (4h Donchian+HMA+RSI+1d): Sharpe=-0.051 but generated trades (kept)
- #1497 (1d Donchian+HMA+RSI+1w): Sharpe=0.424, Return=+91.5% (BEST pattern)
- Complex regime filters (#1501, #1502, #1504, #1507, #1508) ALL failed with 0 trades or negative Sharpe

Key insight: SIMPLICITY WINS. The 0-trade strategies had too many filters (CHOP+CRSI+session+volume+regime).
This strategy uses the proven Donchian+HMA+RSI pattern that worked on 1d, adapted for 4h.

Design:
- 1d HMA(21) for macro trend bias (HTF filter)
- 4h Donchian(20) breakout for entry signal
- 4h RSI(14) filter to avoid chasing overbought/oversold
- 4h HMA(21) for trend confirmation
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.30 (discrete: 0.0, ±0.30)
- LOOSE entry conditions to ensure 20-50 trades/year

Timeframe: 4h (as required by experiment)
HTF: 1d (daily trend bias)
Position Size: 0.30 (discrete levels to minimize fee churn)
Target: 80-200 trades/train (4 years), 20-50 trades/test (15 months), Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d_trend_atr_v2"
timeframe = "4h"
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
    """
    Donchian Channel - breakout system
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 4h (20-50 trades/year target)
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) - confirmation ===
        h4_bull = close[i] > hma_4h[i]
        h4_bear = close[i] < hma_4h[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long breakout: price breaks above Donchian upper
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Short breakout: price breaks below Donchian lower
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER - avoid chasing overbought/oversold ===
        # Long: RSI not overbought (< 70) but showing momentum (> 45)
        rsi_ok_long = 45.0 <= rsi[i] <= 70.0
        # Short: RSI not oversold (> 30) but showing weakness (< 55)
        rsi_ok_short = 30.0 <= rsi[i] <= 55.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h bullish + Donchian breakout + RSI filter
        if daily_bull and h4_bull and donchian_breakout_long and rsi_ok_long:
            desired_signal = BASE_SIZE
        # LONG fallback: 1d bull + 4h bull + RSI momentum (looser for more trades)
        elif daily_bull and h4_bull and rsi[i] > 50.0 and rsi[i] < 65.0:
            desired_signal = BASE_SIZE * 0.8
        # LONG fallback 2: 1d bull + Donchian breakout (trend + breakout)
        elif daily_bull and donchian_breakout_long and rsi[i] < 60.0:
            desired_signal = BASE_SIZE * 0.6
        # LONG fallback 3: 4h bull + Donchian breakout (primary trend only)
        elif h4_bull and donchian_breakout_long and rsi[i] < 65.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT: 1d bearish + 4h bearish + Donchian breakout + RSI filter
        elif daily_bear and h4_bear and donchian_breakout_short and rsi_ok_short:
            desired_signal = -BASE_SIZE
        # SHORT fallback: 1d bear + 4h bear + RSI weakness (looser for more trades)
        elif daily_bear and h4_bear and rsi[i] < 50.0 and rsi[i] > 35.0:
            desired_signal = -BASE_SIZE * 0.8
        # SHORT fallback 2: 1d bear + Donchian breakout (trend + breakout)
        elif daily_bear and donchian_breakout_short and rsi[i] > 40.0:
            desired_signal = -BASE_SIZE * 0.6
        # SHORT fallback 3: 4h bear + Donchian breakout (primary trend only)
        elif h4_bear and donchian_breakout_short and rsi[i] > 35.0:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
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