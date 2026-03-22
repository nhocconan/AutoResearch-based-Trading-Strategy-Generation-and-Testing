#!/usr/bin/env python3
"""
Experiment #520: 1h Primary + 4h/12h HTF — Simplified Multi-TF Trend Following

Hypothesis: After 12 consecutive failures (508-519), the #1 issue is TOO MANY FILTERS = 0 trades.
Strategies 508, 510, 515 all got Sharpe=0.000 (ZERO TRADES). This is the cardinal sin.

NEW APPROACH - SIMPLICITY WINS:
1. 12h HMA = major trend direction (only trade with trend)
2. 4h RSI = pullback detection (enter on dips in uptrend, rallies in downtrend)
3. 1h price action = entry trigger (simple breakout/pullback confirmation)
4. MINIMAL filters - just 2-3 conditions max, NOT 6+ confluence

Why this should work:
- Current best (Sharpe=0.435) uses 1d HMA + 1w HTF — we use 12h + 4h for MORE trades
- 1h TF targets 30-60 trades/year = ~120-240 over 4yr train (well above 30 minimum)
- Fewer conflicting filters = signals actually fire
- Loose RSI thresholds (35/65 not 30/70) ensure entries happen

Position sizing: 0.25 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: >=120 trades/symbol on train, >=10 on test, Sharpe > 0.435
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_simp_4h12h_v1"
timeframe = "1h"
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

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (major trend)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Calculate 4h HTF indicators (pullback detection)
    rsi_4h_14 = calculate_rsi(df_4h['close'].values, period=14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_1h_14 = calculate_rsi(close, 14)
    sma_20 = calculate_sma(close, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1h_14[i]):
            continue
        if np.isnan(sma_20[i]):
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        # Bull: price > HMA21 AND HMA21 > HMA50
        bull_trend = (close[i] > hma_12h_21_aligned[i]) and (hma_12h_21_aligned[i] > hma_12h_50_aligned[i])
        # Bear: price < HMA21 AND HMA21 < HMA50
        bear_trend = (close[i] < hma_12h_21_aligned[i]) and (hma_12h_21_aligned[i] < hma_12h_50_aligned[i])
        
        # === 4H RSI PULLBACK (entry timing) ===
        # Loose thresholds to ensure trades happen
        rsi_4h_oversold = rsi_4h_aligned[i] < 45.0  # Pullback in uptrend
        rsi_4h_overbought = rsi_4h_aligned[i] > 55.0  # Rally in downtrend
        
        # === 1H ENTRY TRIGGERS (simple, frequent) ===
        # Price above/below SMA20 for momentum confirmation
        price_above_sma = close[i] > sma_20[i]
        price_below_sma = close[i] < sma_20[i]
        
        # 1h RSI for additional confirmation (loose thresholds)
        rsi_1h_low = rsi_1h_14[i] < 40.0
        rsi_1h_high = rsi_1h_14[i] > 60.0
        
        # === ENTRY LOGIC — KEEP IT SIMPLE (2-3 conditions max) ===
        new_signal = 0.0
        
        # LONG: 12h bull trend + 4h RSI pullback + 1h momentum
        if bull_trend:
            if rsi_4h_oversold and price_above_sma:
                new_signal = SIZE
            elif rsi_4h_oversold and rsi_1h_low:
                new_signal = SIZE
            elif price_above_sma and rsi_1h_low:
                new_signal = SIZE * 0.8
        
        # SHORT: 12h bear trend + 4h RSI rally + 1h momentum
        if new_signal == 0.0 and bear_trend:
            if rsi_4h_overbought and price_below_sma:
                new_signal = -SIZE
            elif rsi_4h_overbought and rsi_1h_high:
                new_signal = -SIZE
            elif price_below_sma and rsi_1h_high:
                new_signal = -SIZE * 0.8
        
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit long if trend flips bearish
            if bear_trend:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend flips bullish
            if bull_trend:
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
                # Flip position
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