#!/usr/bin/env python3
"""
Experiment #569: 4h Primary + 1d HTF — Asymmetric Trend-Pullback with KAMA

Hypothesis: After 500+ failed experiments, the pattern is clear:
- 4h strategies with dual-regime (chop + connors) are failing badly (#559, #561, #562)
- Simple HMA + RSI pullback also failed (#564 Sharpe=-0.716)
- Key insight: 4h needs STRONGER HTF filter to avoid whipsaws
- Using 1d HMA(21) slope + level as PRIMARY trend filter (not just direction)
- KAMA (Kaufman Adaptive) on 4h adapts to volatility better than HMA/EMA
- Asymmetric entries: ONLY trade in direction of 1d trend (no counter-trend)
- RSI pullback on 4h (not breakout) to catch dips in established trends
- ADX filter relaxed (>18 not >25) to generate sufficient trades
- Target: 20-50 trades/year on 4h per Rule 10

Why this might beat Sharpe=0.435:
- 1d HTF filter is STRONGER than 4h/12h used in failed attempts
- KAMA adapts to market noise (better than static HMA in choppy 2022-2024)
- Asymmetric logic prevents counter-trend losses (major issue in bear markets)
- RSI pullback entries have proven edge (#543, #550 patterns)
- Conservative sizing (0.28) protects against 2022-style crashes

Position sizing: 0.28 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_asymmetric_1d_v1"
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

def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/31):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market noise (Efficiency Ratio).
    Periods: er_period=10, fast_sc=2/(10+1), slow_sc=2/(30+1)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close_s.diff(er_period).values)
    volatility = np.abs(close_s.diff()).values
    
    # Sum of absolute differences over er_period
    vol_sum = pd.Series(volatility).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = vol_sum > 0
    er[mask] = change / vol_sum
    er[:er_period] = np.nan
    
    # Calculate smoothing constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama[:er_period] = np.nan
    return kama

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
    n = len(close)
    
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
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h = calculate_kama(close, er_period=10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
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
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]) or np.isnan(kama_4h[i]):
            continue
        
        # === 1D MAJOR TREND (PRIMARY direction filter - asymmetric) ===
        # Bull regime: price > HMA21 AND HMA21 > HMA50 (trend confirmation)
        bull_regime_1d = (close[i] > hma_1d_21_aligned[i]) and (hma_1d_21_aligned[i] > hma_1d_50_aligned[i])
        # Bear regime: price < HMA21 AND HMA21 < HMA50
        bear_regime_1d = (close[i] < hma_1d_21_aligned[i]) and (hma_1d_21_aligned[i] < hma_1d_50_aligned[i])
        
        # === 4H KAMA TREND (secondary confirmation) ===
        # Price above KAMA = bullish on 4h
        kama_bull_4h = close[i] > kama_4h[i]
        kama_bear_4h = close[i] < kama_4h[i]
        
        # === ADX FILTER (ensure some trend strength) ===
        # ADX > 18 means some directional movement (relaxed from 25)
        trend_ok = adx_14[i] > 18.0
        
        # === RSI PULLBACK ENTRY (asymmetric based on 1d trend) ===
        # LONG: Only in 1d bull regime, RSI 35-55 (pullback, not crash)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        # SHORT: Only in 1d bear regime, RSI 45-65 (rally into resistance)
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC — ASYMMETRIC (only trade with 1d trend) ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d bull + 4h KAMA bull + RSI pullback + ADX OK
        if bull_regime_1d and kama_bull_4h and rsi_pullback_long and trend_ok:
            new_signal = POSITION_SIZE
        
        # SHORT ENTRY: 1d bear + 4h KAMA bear + RSI pullback + ADX OK
        elif bear_regime_1d and kama_bear_4h and rsi_pullback_short and trend_ok:
            new_signal = -POSITION_SIZE
        
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
        
        # === EXIT CONDITIONS (regime flip on 1d) ===
        # Exit long on 1d regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1d:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1d:
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