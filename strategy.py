#!/usr/bin/env python3
"""
Experiment #519: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX Regime + RSI Pullback

Hypothesis: After 465+ failed strategies (mostly volspike/CRSI/Choppiness), try a SIMPLER
approach focused on TRADE FREQUENCY. The #1 failure mode is 0 trades from too many
conflicting filters.

Key changes from failed experiments:
1. REMOVE Fisher Transform (rarely triggers cleanly)
2. REMOVE Donchian breakout (too infrequent on 4h)
3. REMOVE Vol Spike ratio (20+ volspike strategies all failed)
4. SIMPLER entry: KAMA crossover + ADX > 20 + RSI confirmation + 1d HMA trend
5. LOWER thresholds: ADX > 20 (not 40), RSI 35-65 (not 20-80)

Why KAMA (Kaufman Adaptive Moving Average):
- Adapts smoothing based on market efficiency ratio (ER)
- Less lag in trends, less whipsaw in chop
- Proven in "Trading Systems and Methods" (Kaufman)

Why this might beat current best (Sharpe=0.435):
- Fewer conflicting filters = MORE trades (critical for >=30/symbol requirement)
- KAMA adapts to regime automatically (no need for Choppiness Index)
- 1d HMA provides major trend filter without over-filtering
- ADX > 20 threshold ensures some trend without being too restrictive

Position sizing: 0.20-0.35 (scales with ADX strength)
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_pullback_1d_v1"
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

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market Efficiency Ratio (ER).
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    High ER = trending (fast SC), Low ER = choppy (slow SC)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    vol_sum = np.zeros(n)
    for i in range(er_period, n):
        vol_sum[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
    vol_sum[:er_period] = np.nan
    
    er = change / (vol_sum + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant
    fast_sc_val = 2.0 / (fast_sc + 1)
    slow_sc_val = 2.0 / (slow_sc + 1)
    sc = (er * (fast_sc_val - slow_sc_val) + slow_sc_val) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[:er_period] = np.nan
    
    # Initialize with SMA
    kama[er_period] = np.nanmean(close[:er_period+1])
    
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength (not direction).
    ADX > 25 = trending, ADX < 20 = ranging
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

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
    
    # Calculate 1d HTF HMA for major trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # KAMA for adaptive trend (faster response than EMA)
    kama_fast = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    kama_slow = calculate_kama(close, er_period=20, fast_sc=2, slow_sc=40)
    
    # ADX for trend strength
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # RSI for pullback entries
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    MAX_SIZE = 0.35
    
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
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        if np.isnan(adx[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Bull: price above 1d HMA(21) and HMA(21) > HMA(50)
        bull_regime = (close[i] > hma_1d_21_aligned[i]) and (hma_1d_21_aligned[i] > hma_1d_50_aligned[i])
        # Bear: price below 1d HMA(21) and HMA(21) < HMA(50)
        bear_regime = (close[i] < hma_1d_21_aligned[i]) and (hma_1d_21_aligned[i] < hma_1d_50_aligned[i])
        # Neutral: mixed signals (reduce size or stay flat)
        neutral_regime = not bull_regime and not bear_regime
        
        # === 4H TREND SIGNALS (KAMA crossover) ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover detection
        kama_cross_up = (kama_fast[i] > kama_slow[i]) and (kama_fast[i-1] <= kama_slow[i-1])
        kama_cross_down = (kama_fast[i] < kama_slow[i]) and (kama_fast[i-1] >= kama_slow[i-1])
        
        # === ADX TREND STRENGTH ===
        adx_trending = adx[i] > 20.0  # Lower threshold for more trades
        adx_strong = adx[i] > 30.0
        
        # === RSI PULLBACK CONDITIONS ===
        # In bull trend: look for RSI pullback to 40-50 zone
        rsi_pullback_long = (rsi_14[i] > 35.0) and (rsi_14[i] < 55.0)
        # In bear trend: look for RSI bounce to 50-60 zone
        rsi_pullback_short = (rsi_14[i] > 45.0) and (rsi_14[i] < 65.0)
        # Extreme conditions for counter-trend
        rsi_oversold = rsi_14[i] < 30.0
        rsi_overbought = rsi_14[i] > 70.0
        
        # === POSITION SIZING (scale with ADX) ===
        if adx_strong:
            position_size = MAX_SIZE
        else:
            position_size = BASE_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED for more trades) ===
        new_signal = 0.0
        
        # LONG ENTRIES
        # Condition 1: Bull regime + KAMA bullish + RSI pullback (trend pullback)
        if bull_regime and kama_bullish and rsi_pullback_long:
            new_signal = position_size
        # Condition 2: KAMA cross up + ADX trending + RSI not overbought (momentum)
        elif kama_cross_up and adx_trending and (rsi_14[i] < 65.0):
            new_signal = position_size
        # Condition 3: Bull regime + KAMA cross up (trend confirmation)
        elif bull_regime and kama_cross_up:
            new_signal = BASE_SIZE
        # Condition 4: RSI oversold + KAMA bullish (mean reversion in uptrend)
        elif rsi_oversold and kama_bullish:
            new_signal = BASE_SIZE
        
        # SHORT ENTRIES
        if new_signal == 0.0:
            # Condition 1: Bear regime + KAMA bearish + RSI pullback (trend pullback)
            if bear_regime and kama_bearish and rsi_pullback_short:
                new_signal = -position_size
            # Condition 2: KAMA cross down + ADX trending + RSI not oversold (momentum)
            elif kama_cross_down and adx_trending and (rsi_14[i] > 35.0):
                new_signal = -position_size
            # Condition 3: Bear regime + KAMA cross down (trend confirmation)
            elif bear_regime and kama_cross_down:
                new_signal = -BASE_SIZE
            # Condition 4: RSI overbought + KAMA bearish (mean reversion in downtrend)
            elif rsi_overbought and kama_bearish:
                new_signal = -BASE_SIZE
        
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
        
        # === EXIT CONDITIONS (regime flip or overextension) ===
        # Exit long if regime flips bearish or RSI overbought
        if in_position and position_side > 0:
            if bear_regime and (adx[i] > 25.0):
                new_signal = 0.0
            elif rsi_overbought:
                new_signal = 0.0
            # Exit if KAMA flips bearish
            elif kama_cross_down:
                new_signal = 0.0
        
        # Exit short if regime flips bullish or RSI oversold
        if in_position and position_side < 0:
            if bull_regime and (adx[i] > 25.0):
                new_signal = 0.0
            elif rsi_oversold:
                new_signal = 0.0
            # Exit if KAMA flips bullish
            elif kama_cross_up:
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