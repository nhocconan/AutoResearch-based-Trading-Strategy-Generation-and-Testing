#!/usr/bin/env python3
"""
Experiment #129: 4h Primary + 1d HTF — HMA Trend + Donchian Breakout + Loose RSI

Hypothesis: After analyzing 100+ failed experiments, the winning pattern is clear:
- HMA (Hull Moving Average) is more responsive than KAMA/EMA for crypto trends
- Donchian breakout (20-period) provides clean entry signals without overfitting
- 4h timeframe with 1d HTF bias has proven success (SOL +0.879 with HMA)
- VERY loose RSI thresholds (20/80) ensure trade generation on ALL symbols
- Simple logic > complex regime detection (complex = 0 trades or negative Sharpe)

This strategy uses MINIMAL but effective filters:
1. 1d HMA = major trend bias (price above/below)
2. 4h Donchian(20) breakout = entry trigger (clean, proven signal)
3. 4h HMA(21) = trend confirmation (price above HMA for long, below for short)
4. RSI loose filter (>20 for long, <80 for short) - ensures trades generate
5. ATR trailing stoploss (2.5x) for risk management
6. NO Choppiness, NO complex regime detection, NO Fisher Transform

Key design choices:
- Timeframe: 4h (proven to work, 20-50 trades/year target)
- HTF: 1d for trend bias (responsive enough, not too slow like 1w)
- HMA: eliminates lag better than EMA/KAMA for crypto momentum
- Donchian(20): classic breakout, works in both trending and ranging markets
- RSI thresholds: 20/80 (very loose, ensures trades on BTC/ETH/SOL)
- Position size: 0.30 (30% of capital, conservative for 4h)
- Stoploss: 2.5x ATR trailing (proven in baseline strategy)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_rsi_loose_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    Eliminates lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.zeros(len(series))
        result[:] = np.nan
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i-span+1:i+1] * weights)
        return result
    
    close_series = np.array(close)
    wma_half = wma(close_series, period // 2)
    wma_full = wma(close_series, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    diff = 2 * wma_half - wma_full
    hma = wma(diff, int(np.sqrt(period)))
    
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 4h)
    
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
        if np.isnan(hma_4h[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        # Simple: is price above or below daily HMA?
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA) ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks above Donchian upper (20-period high)
        # Short: price breaks below Donchian lower (20-period low)
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # break below previous lower
        
        # === RSI FILTER (VERY LOOSE - ensure trades generate on all symbols) ===
        # For longs: RSI > 20 (not extremely oversold)
        # For shorts: RSI < 80 (not extremely overbought)
        rsi_ok_long = rsi[i] > 20.0
        rsi_ok_short = rsi[i] < 80.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h HMA bull + Donchian breakout + RSI > 20
        # SHORT: 1d bear + 4h HMA bear + Donchian breakout + RSI < 80
        desired_signal = 0.0
        
        if htf_bull and hma_bull and donchian_breakout_long and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and hma_bear and donchian_breakout_short and rsi_ok_short:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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