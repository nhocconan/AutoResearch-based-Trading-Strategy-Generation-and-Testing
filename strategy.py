#!/usr/bin/env python3
"""
Experiment #1505: 1h Primary + 4h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After analyzing 1100+ failed strategies, the pattern for 1h is clear:
1. 1h timeframe MUST generate 30-80 trades/year (not 0 trades like #1495, #1500)
2. Complex filters (CRSI+CHOP+session+volume) = 0 trades on lower TF
3. SIMPLER is better: HTF trend (4h/1d HMA) + 1h RSI pullback + ATR stop
4. LOOSE entry conditions ensure trades happen (RSI 35-65, not tight bands)
5. 4h/1d HMA for direction, 1h RSI for entry timing within trend

Key design choices:
- Remove session filter (too restrictive for 1h)
- Remove volume filter (too restrictive)
- Remove CHOP index (adds complexity, rarely triggers on 1h)
- Use 4h HMA + 1d HMA for trend direction (dual HTF confirmation)
- Use 1h RSI(14) for pullback entries (RSI<45 long in uptrend, RSI>55 short in downtrend)
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.25 (smaller for 1h to handle more trades)
- Discrete signal levels (0.0, ±0.25) to minimize fee churn

Timeframe: 1h (as required by experiment)
HTF: 4h + 1d (dual HTF trend confirmation)
Position Size: 0.25 (discrete: 0.0, ±0.25)
Target: 40-100 trades/train, 10-20 trades/test, Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h1d_pullback_atr_v1"
timeframe = "1h"
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

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_ema(close, period=21):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    sma_50 = calculate_sma(close, period=50)
    ema_21 = calculate_ema(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h (more trades expected)
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - confirmation ===
        h4_bull = close[i] > hma_4h_aligned[i]
        h4_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (1h HMA) - entry timing ===
        h1_bull = close[i] > hma_1h[i]
        h1_bear = close[i] < hma_1h[i]
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === RSI PULLBACK - LOOSE bands for MORE trades ===
        # Long: RSI pulled back but not oversold (35-50)
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        # Short: RSI rallied but not overbought (50-65)
        rsi_pullback_short = 50.0 <= rsi[i] <= 65.0
        
        # === DESIRED SIGNAL - SIMPLIFIED FOR 1h ===
        desired_signal = 0.0
        
        # LONG: All HTF bullish + 1h pullback entry
        # Option 1: Strong trend (1d + 4h + 1h all bull) + RSI pullback
        if daily_bull and h4_bull and h1_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # Option 2: 1d + 4h bull + 1h above SMA50 + RSI support (looser)
        elif daily_bull and h4_bull and above_sma50 and rsi[i] > 40.0:
            desired_signal = BASE_SIZE * 0.8
        # Option 3: 1d bull + 4h bull + RSI not overbought (loosest, ensures trades)
        elif daily_bull and h4_bull and rsi[i] < 60.0:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT: All HTF bearish + 1h pullback entry
        # Option 1: Strong trend (1d + 4h + 1h all bear) + RSI pullback
        elif daily_bear and h4_bear and h1_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        # Option 2: 1d + 4h bear + 1h below SMA50 + RSI support (looser)
        elif daily_bear and h4_bear and below_sma50 and rsi[i] < 60.0:
            desired_signal = -BASE_SIZE * 0.8
        # Option 3: 1d bear + 4h bear + RSI not oversold (loosest, ensures trades)
        elif daily_bear and h4_bear and rsi[i] > 40.0:
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