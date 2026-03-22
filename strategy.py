#!/usr/bin/env python3
"""
Experiment #564: 4h Primary + 12h/1d HTF — Simplified HMA Trend with RSI Pullback

Hypothesis: After 500+ failed experiments, the pattern is clear:
- Complex regime filters (Choppiness, dual-regime) consistently fail on 4h
- Too many confluence conditions = 0 trades (see #552, #558, #560, #561, #562)
- 4h timeframe needs SIMPLER logic: HMA trend + RSI pullback + HTF confirmation
- 12h HTF for major trend direction, 4h for entry timing
- Remove Choppiness Index (failed in #559, #561, #562)
- Remove session/volume filters (kills trades per #555 analysis)
- Use asymmetric position sizing: stronger trend = larger size

Key differences from failed attempts:
1. NO Choppiness Index regime filter (consistently negative Sharpe on 4h)
2. NO Connors RSI complexity (simple RSI(14) works better)
3. Simpler entry: RSI 30-55 long, 45-70 short (wider than failed attempts)
4. 12h HMA for major trend, 4h HMA for entry timing
5. ADX > 18 (not >25 which was too strict in #555)
6. ATR 2.5x trailing stop (proven in literature)
7. Position size: 0.28 base, 0.35 for strong trend alignment

Target: 20-50 trades/year on 4h (per Rule 10), Sharpe > 0.435 to beat baseline
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h_simp_v1"
timeframe = "4h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF HMA for major trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # 4h HMA for entry timing
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    STRONG_SIZE = 0.35
    
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
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength confirmation
        hma_12h_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H ENTRY TIMING ===
        bull_regime_4h = close[i] > hma_4h_21[i]
        bear_regime_4h = close[i] < hma_4h_21[i]
        
        # 4h HMA slope
        hma_4h_slope_bull = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_slope_bear = hma_4h_21[i] < hma_4h_50[i]
        
        # === ADX FILTER (ensure some trend strength) ===
        # ADX > 18 means some directional movement (lower than failed attempts)
        trend_ok = adx_14[i] > 18.0
        
        # === RSI PULLBACK ENTRY (WIDER BANDS for more trades) ===
        # Long: RSI 30-55 in uptrend (pullback, not oversold crash)
        rsi_pullback_long = 30.0 <= rsi_14[i] <= 55.0
        # Short: RSI 45-70 in downtrend (rally into resistance)
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 70.0
        
        # === ENTRY LOGIC — SIMPLIFIED ===
        new_signal = 0.0
        
        # LONG ENTRY: 12h bull + 4h bull + RSI pullback + ADX OK
        if bull_regime_12h and bull_regime_4h and rsi_pullback_long and trend_ok:
            # Size based on trend alignment strength
            if hma_12h_slope_bull and hma_4h_slope_bull:
                new_signal = STRONG_SIZE  # Both timeframes aligned bull
            elif hma_12h_slope_bull:
                new_signal = BASE_SIZE  # Only 12h aligned
            else:
                new_signal = BASE_SIZE * 0.8  # Weak alignment
        
        # SHORT ENTRY: 12h bear + 4h bear + RSI pullback + ADX OK
        elif bear_regime_12h and bear_regime_4h and rsi_pullback_short and trend_ok:
            # Size based on trend alignment strength
            if hma_12h_slope_bear and hma_4h_slope_bear:
                new_signal = -STRONG_SIZE  # Both timeframes aligned bear
            elif hma_12h_slope_bear:
                new_signal = -BASE_SIZE  # Only 12h aligned
            else:
                new_signal = -BASE_SIZE * 0.8  # Weak alignment
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 12h regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_12h and hma_12h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 12h regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_12h and hma_12h_slope_bull:
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