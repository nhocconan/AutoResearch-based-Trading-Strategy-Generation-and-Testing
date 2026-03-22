#!/usr/bin/env python3
"""
Experiment #396: 12h Primary + 1d HTF — HMA Trend + RSI Pullback + ATR Stop

Hypothesis: After 358 failed experiments, the pattern is clear:
1. Complex regime-switching (Choppiness, dual-regime) FAILS on 12h/1d (see #386, #392, #393)
2. SIMPLE trend + pullback works best (current best: mtf_1d_hma_rsi_1w_simp_v2, Sharpe=0.435)
3. 12h timeframe should generate 20-50 trades/year (~1-2 trades/week)
4. 1d HMA(21) for major trend filter (proven in #382, #389)
5. 12h HMA(8/21) crossover for faster entry signals than HMA(16/48)
6. RSI(14) pullback with WIDER range (35-65) to ensure trade frequency across ALL symbols
7. ATR 2.5x trailing stop for risk management

Why this might beat Sharpe=0.435:
- 12h captures cleaner trends than 4h (less noise, fewer false breakouts)
- HMA(8/21) faster than HMA(16/48) — enters trends earlier on 12h
- Wider RSI range ensures >=30 trades/symbol on train (critical requirement)
- 1d HTF filter prevents counter-trend trades (reduces whipsaw in 2022 crash)
- Discrete position sizing minimizes fee churn

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 20-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d_v1"
timeframe = "12h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_8 = calculate_hma(close, period=8)
    hma_12h_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_8[i]) or np.isnan(hma_12h_21[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d HMA = bull market bias (favor longs)
        # Price below 1d HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_12h_8[i] > hma_12h_21[i]
        hma_bearish = hma_12h_8[i] < hma_12h_21[i]
        
        # === RSI PULLBACK SIGNALS (wider range for trade frequency) ===
        # Long: RSI pulled back to 35-60 in uptrend (buying dip)
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 60.0
        # Short: RSI pulled back to 40-65 in downtrend (selling rally)
        rsi_short_pullback = 40.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC — SIMPLE TREND + PULLBACK ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + HMA bullish + RSI pullback
        if bull_regime and hma_bullish and rsi_long_pullback:
            new_signal = LONG_SIZE
        
        # SHORT ENTRY: Bear regime + HMA bearish + RSI pullback
        if bear_regime and hma_bearish and rsi_short_pullback:
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 8 bars (~4 days on 12h), force entry on weaker signal
        if bars_since_last_trade > 8 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_14[i] < 55 and hma_bullish:
                new_signal = LONG_SIZE * 0.8
            elif bear_regime and rsi_14[i] > 45 and hma_bearish:
                new_signal = -SHORT_SIZE * 0.8
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (12h HMA cross)
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals