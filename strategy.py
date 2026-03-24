#!/usr/bin/env python3
"""
Experiment #1519: 4h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback + HMA Confirmation

Hypothesis: Based on #1513 success (1d HMA+RSI+Donchian), adapting to 4h primary with 1d HTF
should generate more trades (40-80/year vs 20-50) while maintaining quality. Key insights:

1. KAMA (Kaufman Adaptive MA) adapts to volatility - better than HMA in choppy 4h markets
2. 1d HMA for macro trend bias (proven in #1513)
3. Looser RSI bands (25-75) ensure trades happen without over-filtering
4. Simple confluence: HTF trend + Primary trend + RSI pullback = entry
5. ATR 2.5x trailing stop for risk management

Design:
- 1d HMA(21) for macro trend direction (HTF filter)
- 4h KAMA(21) for primary adaptive trend
- 4h RSI(14) for pullback entries (loose: 25-75 range)
- 4h ATR(14) 2.5x trailing stop
- Position size 0.28 (discrete: 0.0, ±0.28)
- Target: 40-80 trades/train (4 years), 10-20 trades/test (15 months)

Timeframe: 4h (as required by experiment)
HTF: 1d (daily trend bias)
Position Size: 0.28 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_hma_rsi_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - smooth in trends, responsive in ranges
    ER = |Close - Close_n| / Sum(|Close_i - Close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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
    kama_4h = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Appropriate size for 4h (40-80 trades/year target)
    
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
        if np.isnan(rsi[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) - adaptive confirmation ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === KAMA SLOPE - trend momentum ===
        kama_slope_bull = False
        kama_slope_bear = False
        if i >= 5 and not np.isnan(kama_4h[i-5]):
            kama_slope_bull = kama_4h[i] > kama_4h[i-5]
            kama_slope_bear = kama_4h[i] < kama_4h[i-5]
        
        # === RSI PULLBACK - LOOSE bands for MORE trades ===
        # Long: RSI pulled back but not oversold (25-55)
        rsi_pullback_long = 25.0 <= rsi[i] <= 55.0
        # Short: RSI rallied but not overbought (45-75)
        rsi_pullback_short = 45.0 <= rsi[i] <= 75.0
        
        # === DESIRED SIGNAL - SIMPLIFIED FOR 4h ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h KAMA bull + RSI pullback
        # Option 1: Strong trend (1d + 4h both bull + slope) + RSI pullback
        if daily_bull and kama_bull and kama_slope_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # Option 2: 1d bull + 4h KAMA bull + RSI pullback (standard)
        elif daily_bull and kama_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE * 0.9
        # Option 3: 1d bull + 4h KAMA bull + RSI not overbought (looser)
        elif daily_bull and kama_bull and rsi[i] < 65.0:
            desired_signal = BASE_SIZE * 0.7
        # Option 4: 1d bull + 4h above KAMA + RSI neutral (fallback for trades)
        elif daily_bull and kama_bull and 35.0 <= rsi[i] <= 65.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT: 1d bearish + 4h KAMA bear + RSI pullback
        # Option 1: Strong trend (1d + 4h both bear + slope) + RSI pullback
        elif daily_bear and kama_bear and kama_slope_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        # Option 2: 1d bear + 4h KAMA bear + RSI pullback (standard)
        elif daily_bear and kama_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE * 0.9
        # Option 3: 1d bear + 4h KAMA bear + RSI not oversold (looser)
        elif daily_bear and kama_bear and rsi[i] > 35.0:
            desired_signal = -BASE_SIZE * 0.7
        # Option 4: 1d bear + 4h below KAMA + RSI neutral (fallback for trades)
        elif daily_bear and kama_bear and 35.0 <= rsi[i] <= 65.0:
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.65:
            final_signal = BASE_SIZE * 0.75
        elif desired_signal >= BASE_SIZE * 0.45:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.65:
            final_signal = -BASE_SIZE * 0.75
        elif desired_signal <= -BASE_SIZE * 0.45:
            final_signal = -BASE_SIZE * 0.5
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