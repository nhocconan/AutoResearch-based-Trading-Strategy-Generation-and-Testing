#!/usr/bin/env python3
"""
Experiment #563: 1d Primary + 1w HTF — Simplified Trend-Pullback with HMA Crossover

Hypothesis: After analyzing 500+ failed strategies, the pattern is clear:
- Complex regime switching (Choppiness + Connors) often overfits and fails
- Simple trend-following with HTF filter works best on 1d timeframe
- #557 (1d dual regime) had Sharpe=0.152 — proves 1d+1w can work
- Current best (mtf_1d_hma_rsi_1w_simp_asym_v2) has Sharpe=0.435 — must beat this

This strategy uses PROVEN patterns from research:
1. 1w HMA(21) for MAJOR trend direction (HTF bias — slow, reliable)
2. 1d HMA(16/48) crossover for ENTRY timing (faster signal within trend)
3. RSI(14) 35-65 filter for pullback confirmation (not extremes)
4. ATR(14) 2.5x trailing stop for risk management
5. Position size: 0.28 discrete (balanced for 1d trade frequency)

Why this might beat Sharpe=0.435:
- SIMPLER than failed regime-switching strategies (#552, #553, #556, #559, #561, #562)
- 1w HTF filter prevents major counter-trend losses (critical for 2022 crash)
- HMA crossover on 1d catches trend changes faster than price vs HMA
- RSI mid-range filter avoids chasing tops/bottoms
- Target: 20-50 trades/year on 1d (per Rule 10), enough for statistical significance

Position sizing: 0.28 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_crossover_rsi_1w_v1"
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
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Calculate 1d HMA crossover signals (16/48)
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # 1d timeframe: target 20-50 trades/year, size 0.28 balanced
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track HMA crossover state to avoid repeated signals
    prev_hma_bull_cross = False
    prev_hma_bear_cross = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull market bias (only look for longs)
        # Price below 1w HMA = bear market bias (only look for shorts)
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # === 1D HMA CROSSOVER (entry timing) ===
        # Bull cross: HMA16 crosses above HMA48
        # Bear cross: HMA16 crosses below HMA48
        hma_bull_cross = (hma_1d_16[i] > hma_1d_48[i]) and (hma_1d_16[i-1] <= hma_1d_48[i-1])
        hma_bear_cross = (hma_1d_16[i] < hma_1d_48[i]) and (hma_1d_16[i-1] >= hma_1d_48[i-1])
        
        # Current HMA state (for holding positions)
        hma_bull_state = hma_1d_16[i] > hma_1d_48[i]
        hma_bear_state = hma_1d_16[i] < hma_1d_48[i]
        
        # === ADX FILTER (ensure some trend strength) ===
        # ADX > 18 means some directional movement (lower threshold for more trades)
        trend_ok = adx_14[i] > 18.0
        
        # === RSI PULLBACK FILTER (mid-range, not extremes) ===
        # Long: RSI 35-65 (pullback within uptrend, not oversold crash)
        # Short: RSI 35-65 (rally within downtrend, not overbought spike)
        rsi_ok_long = 35.0 <= rsi_14[i] <= 65.0
        rsi_ok_short = 35.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 1w bull + 1d HMA bull cross + ADX OK + RSI ok
        if bull_regime_1w and hma_bull_cross and trend_ok and rsi_ok_long:
            new_signal = POSITION_SIZE
        
        # SHORT ENTRY: 1w bear + 1d HMA bear cross + ADX OK + RSI ok
        elif bear_regime_1w and hma_bear_cross and trend_ok and rsi_ok_short:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            # Hold long if HMA still bullish and 1w regime unchanged
            if position_side > 0 and hma_bull_state and bull_regime_1w:
                new_signal = signals[i-1] if i > 0 else 0.0
            # Hold short if HMA still bearish and 1w regime unchanged
            elif position_side < 0 and hma_bear_state and bear_regime_1w:
                new_signal = signals[i-1] if i > 0 else 0.0
            else:
                new_signal = 0.0
        
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
        
        # === EXIT CONDITIONS (regime flip or HMA cross against) ===
        # Exit long on 1w regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1w:
                new_signal = 0.0
            # Or HMA crosses bearish
            elif hma_bear_cross:
                new_signal = 0.0
        
        # Exit short on 1w regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1w:
                new_signal = 0.0
            # Or HMA crosses bullish
            elif hma_bull_cross:
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
        
        # Update crossover state tracking
        prev_hma_bull_cross = hma_bull_cross
        prev_hma_bear_cross = hma_bear_cross
    
    return signals