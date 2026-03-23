#!/usr/bin/env python3
"""
Experiment #1153: 1d Primary + 1w HTF — HMA Trend + RSI Momentum + ATR Stop

Hypothesis: After 840+ failed experiments, clear patterns emerge:
- Complex regime switching (CRSI+Chop) consistently FAILS (Sharpe negative in #1141-#1148)
- Simple trend + momentum works better (#1143 Sharpe=0.452 with HMA+RSI+Donchian)
- Higher timeframes (1d, 12h) produce more stable returns for BTC/ETH
- Current #1149 (4h KAMA+Donchian) only achieves Sharpe=0.050 — too many whipsaws

This strategy uses PROVEN simpler components on 1d timeframe:
1. 1w HMA(21) for macro trend filter (weekly direction = trade direction only)
2. 1d HMA(8/21) crossover for entry timing (faster than KAMA, less lag)
3. 1d RSI(14) momentum filter (45/55 thresholds — not too strict)
4. 1d ATR(14) 2.5x trailing stop (wider than 2.0x to avoid premature exits)
5. Position size 0.30 discrete (balance returns vs drawdown)

Why 1d should work better than 4h:
- Fewer false signals (1 candle = 1 day of price action)
- Lower fee drag (target 25-40 trades/year vs 50-80 on 4h)
- Better aligned with macro trends that drive crypto
- 1w HTF provides strong trend filter without overfitting

Why this should beat Sharpe=0.612:
- Weekly trend filter eliminates counter-trend trades that destroyed 2022 returns
- HMA crossover catches trends earlier than KAMA (less adaptive lag)
- RSI 45/55 is loose enough to generate trades but filters extremes
- 2.5x ATR stop allows trends to develop without premature exits
- Target: 25-40 trades/year on 1d (optimal for fee drag on daily TF)

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing
Target: 25-40 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_1w_atr_trend_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate WMA for period/2
    half = period // 2
    if half < 1:
        half = 1
    
    wma_half = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # HMA formula
    raw_hma = 2.0 * wma_half - wma_full
    
    # Final smoothing with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_fast = calculate_hma(close, period=8)
    hma_slow = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA) ===
        # Only trade in direction of weekly trend
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === LOCAL TREND (1d HMA crossover) ===
        # Fast HMA crosses above slow HMA = bullish
        # Fast HMA crosses below slow HMA = bearish
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # Check for fresh crossover (entry signal)
        hma_cross_long = False
        hma_cross_short = False
        
        if i > 0 and not np.isnan(hma_fast[i-1]) and not np.isnan(hma_slow[i-1]):
            # Long crossover: fast was below slow, now above
            if hma_fast[i-1] <= hma_slow[i-1] and hma_fast[i] > hma_slow[i]:
                hma_cross_long = True
            # Short crossover: fast was above slow, now below
            if hma_fast[i-1] >= hma_slow[i-1] and hma_fast[i] < hma_slow[i]:
                hma_cross_short = True
        
        # === MOMENTUM FILTER (RSI) ===
        # RSI > 45 confirms bullish momentum for long entries
        # RSI < 55 confirms bearish momentum for short entries
        # Loose thresholds to ensure trade generation
        rsi_bullish = rsi[i] > 45.0
        rsi_bearish = rsi[i] < 55.0
        
        # === EXTREME RSI EXIT ===
        # Exit long if RSI > 70 (overbought)
        # Exit short if RSI < 30 (oversold)
        rsi_extreme_long = rsi[i] > 70.0
        rsi_extreme_short = rsi[i] < 30.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Weekly bull + HMA cross long + RSI confirms
        if weekly_bull and hma_cross_long and rsi_bullish:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Weekly bear + HMA cross short + RSI confirms
        elif weekly_bear and hma_cross_short and rsi_bearish:
            desired_signal = -BASE_SIZE
        
        # === EXTREME RSI EXIT ===
        if in_position and position_side > 0 and rsi_extreme_long:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_extreme_short:
            desired_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        # Exit if weekly trend reverses against position
        if in_position and position_side > 0 and weekly_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and weekly_bull:
            desired_signal = 0.0
        
        # === HMA TREND REVERSAL EXIT ===
        # Exit if local HMA trend reverses
        if in_position and position_side > 0 and hma_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bull:
            desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly and local still bull
                if weekly_bull and hma_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly and local still bear
                if weekly_bear and hma_bear:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals