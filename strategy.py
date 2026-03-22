#!/usr/bin/env python3
"""
Experiment #514: 4h Primary + 12h HTF — KAMA Adaptive Trend + ADX Regime + RSI Entry

Hypothesis: After 448+ failed strategies (mostly CRSI/Choppiness/VolSpike combos), 
try a SIMPLER approach with fewer conflicting filters to ensure trade frequency.

PROBLEM WITH PREVIOUS STRATEGIES:
- Too many entry conditions (6+ confluence) = 0 trades (experiments 505, 506, 508, 510)
- Complex regime switches blocking entries
- Vol spike conditions too rare for consistent signals

NEW APPROACH:
1. KAMA (Kaufman Adaptive Moving Average) - adapts to market noise better than HMA/EMA
   Works well in both trending and choppy markets. Proven in quant literature.
   
2. ADX regime detection with LOWER threshold (20 instead of 25) - more trades
   ADX > 20 = trend (follow KAMA direction)
   ADX < 20 = range (mean revert at RSI extremes)
   
3. RSI(7) for entry timing - faster than RSI(14), catches more reversals
   Long: RSI < 35 in bull regime OR RSI < 25 in any regime
   Short: RSI > 65 in bear regime OR RSI > 75 in any regime
   
4. 12h HMA for major trend (not 1d - more responsive, still filters noise)
   Price > 12h HMA = bull bias (prefer longs)
   Price < 12h HMA = bear bias (prefer shorts)

5. SIMPLIFIED exit logic - no complex regime flip exits that block trades
   Only exit on: stoploss hit OR opposite signal generated

Why this might beat current best (Sharpe=0.435):
- FEWER filters = MORE trades (critical: need >=30/symbol on train)
- KAMA adapts better than HMA in crypto's choppy markets
- Lower ADX threshold (20 vs 25) = 30-50% more trade opportunities
- RSI(7) faster than RSI(14) = catches more reversals
- 4h TF targets 20-50 trades/year (lower fee drag than 1h/30m)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 25-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi7_12h_simp_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    ER_period: Efficiency Ratio lookback
    Fast_period: Fast SC smoothing constant (trend)
    Slow_period: Slow SC smoothing constant (noise)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio: |change| / sum(|changes|)
    change = np.abs(close_s.diff(er_period))
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period (7 for faster signals)."""
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
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # KAMA adaptive trend (primary signal)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    
    # ADX for regime detection (lower threshold = more trades)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # RSI(7) for entry timing (faster than RSI(14))
    rsi_7 = calculate_rsi(close, period=7)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.30
    
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
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        if np.isnan(kama_21[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_7[i]):
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_12h_21_aligned[i]
        bear_regime = close[i] < hma_12h_21_aligned[i]
        
        # === ADX REGIME DETECTION (trend vs range) ===
        trending = adx_14[i] > 20.0  # Lower threshold for more trades
        ranging = adx_14[i] <= 20.0
        
        # === KAMA TREND DIRECTION ===
        kama_bull = kama_21[i] > kama_50[i]
        kama_bear = kama_21[i] < kama_50[i]
        
        # === RSI EXTREMES (faster with period=7) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_low = rsi_7[i] < 25.0
        rsi_extreme_high = rsi_7[i] > 75.0
        
        # === ENTRY LOGIC — SIMPLIFIED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (2 main conditions - easier to trigger)
        # Condition 1: Bull regime + RSI oversold (pullback in uptrend)
        if bull_regime and rsi_oversold:
            new_signal = LONG_SIZE
        # Condition 2: Extreme RSI (any regime - capitulation)
        elif rsi_extreme_low:
            new_signal = LONG_SIZE
        # Condition 3: Trending + KAMA bull + RSI not overbought
        elif trending and kama_bull and rsi_7[i] < 60.0:
            new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: Bear regime + RSI overbought (bounce in downtrend)
            if bear_regime and rsi_overbought:
                new_signal = -SHORT_SIZE
            # Condition 2: Extreme RSI (any regime - FOMO top)
            elif rsi_extreme_high:
                new_signal = -SHORT_SIZE
            # Condition 3: Trending + KAMA bear + RSI not oversold
            elif trending and kama_bear and rsi_7[i] > 40.0:
                new_signal = -SHORT_SIZE * 0.8
        
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