#!/usr/bin/env python3
"""
Experiment #493: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 492 experiments, clear patterns emerge for 1d timeframe:
1. 1d primary with 1w HTF trend filter worked in research (SOL Sharpe +0.782)
2. Donchian(20) breakouts capture sustained moves without whipsaw
3. HMA(21) on 1w provides clean major trend direction
4. RSI(14) filter prevents chasing overextended breakouts
5. Simpler logic = more trades (critical: need >=30 trades/symbol on train)

Why this might beat current best (Sharpe=0.435):
- Donchian breakouts work in both bull and bear markets
- 1w HMA filter avoids counter-trend trades that destroyed 2022 performance
- RSI filter (30-70 range) ensures entries aren't at extremes
- ATR 2.5x trailing stop protects in crash scenarios
- 1d has minimal fee drag (~10-20 trades/year expected)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 15-30 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_1w_rsi_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_keltner_channels(high, low, close, atr_period=14, mult=2.0):
    """Calculate Keltner Channels for volatility-based bands."""
    atr = calculate_atr(high, low, close, atr_period)
    
    ema_close = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    upper = ema_close + mult * atr
    lower = ema_close - mult * atr
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian_channels(high, low, period=20)
    keltner_upper, keltner_lower = calculate_keltner_channels(high, low, close, atr_period=14, mult=2.0)
    
    # HMA on 1d for local trend
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === RSI FILTER (avoid extremes) ===
        rsi_neutral = 30.0 < rsi_14[i] < 70.0
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Break above upper band = bullish breakout
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Break below lower band = bearish breakout
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === KELTNER SQUEEZE DETECTION ===
        # When price is inside Keltner, volatility is low (potential breakout setup)
        inside_keltner = (close[i] < keltner_upper[i]) and (close[i] > keltner_lower[i])
        
        # === ENTRY LOGIC — DONCHIAN BREAKOUT WITH FILTERS ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence conditions for quality)
        if bull_regime and donchian_breakout_long and rsi_neutral:
            new_signal = LONG_SIZE
        elif bull_regime and hma_bullish and donchian_breakout_long:
            new_signal = LONG_SIZE
        elif hma_bullish and donchian_breakout_long and rsi_oversold:
            new_signal = LONG_SIZE * 0.8
        elif bull_regime and hma_bullish and close[i] > donchian_middle[i] and rsi_14[i] < 50.0:
            new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (multiple confluence conditions for quality)
        if new_signal == 0.0:
            if bear_regime and donchian_breakout_short and rsi_neutral:
                new_signal = -SHORT_SIZE
            elif bear_regime and hma_bearish and donchian_breakout_short:
                new_signal = -SHORT_SIZE
            elif hma_bearish and donchian_breakout_short and rsi_overbought:
                new_signal = -SHORT_SIZE * 0.8
            elif bear_regime and hma_bearish and close[i] < donchian_middle[i] and rsi_14[i] > 50.0:
                new_signal = -SHORT_SIZE * 0.7
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long when RSI overbought
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        # Exit short when RSI oversold
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit (major trend reversal)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals