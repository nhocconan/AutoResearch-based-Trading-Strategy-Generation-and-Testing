#!/usr/bin/env python3
"""
Experiment #491: 4h Primary + 1d/1w HTF — KAMA + Fisher Transform + ADX Regime

Hypothesis: After 490 experiments, clear patterns emerge:
1. 4h has potential but needs cleaner signals (too many CRSI attempts failed)
2. Fisher Transform catches reversals better than RSI in bear/range markets
3. KAMA adapts to volatility better than HMA/EMA (less whipsaw in 2022)
4. ADX regime switch: ADX<20 = mean revert, ADX>25 = trend follow
5. 1d HMA provides major trend bias without over-filtering

Why this might beat current best (Sharpe=0.435):
- Fisher Transform is fundamentally different from RSI/CRSI (new signal type)
- KAMA efficiency ratio adapts to market conditions automatically
- Simpler entry logic = more trades (critical for >=30 trades/symbol)
- Dual regime captures both trending and ranging markets
- 4h has proven potential when filters are not too strict

Position sizing: 0.30 long, 0.25 short (discrete, max 0.40)
Stoploss: 2.0 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_adx_regime_1d_v1"
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
    KAMA adapts to market noise via Efficiency Ratio.
    ER near 1 = trending (fast smoothing), ER near 0 = choppy (slow smoothing)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio: net change / sum of absolute changes
    change = np.abs(close_s.diff())
    noise = change.rolling(window=er_period, min_periods=er_period).sum()
    net_change = np.abs(close_s.diff(er_period))
    
    er = net_change / (noise + 1e-10)
    er = er.clip(0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s) / 2.0
    
    # Normalize to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, 1e-10)
    
    normalized = 2.0 * (typical - lowest) / range_hl - 1.0
    normalized = normalized.clip(-0.999, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    
    # Signal line (1-period lag)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
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
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    adx_4h, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(fisher[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === ADX REGIME DETECTION ===
        is_trending = adx_4h[i] > 25.0
        is_ranging = adx_4h[i] < 20.0
        
        # === KAMA TREND (4h local) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -1.0
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 1.0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # LONG ENTRIES
        if bull_regime and kama_bullish and fisher_oversold:
            new_signal = LONG_SIZE
        elif bull_regime and is_ranging and rsi_oversold and fisher_cross_up:
            new_signal = LONG_SIZE
        elif is_trending and kama_bullish and fisher_cross_up and adx_4h[i] > 20:
            new_signal = LONG_SIZE
        elif kama_bullish and fisher[i] < -1.0 and rsi_14[i] < 40:
            new_signal = LONG_SIZE * 0.8
        elif bull_regime and rsi_oversold and close[i] > kama_4h[i]:
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES
        if new_signal == 0.0:
            if bear_regime and kama_bearish and fisher_overbought:
                new_signal = -SHORT_SIZE
            elif bear_regime and is_ranging and rsi_overbought and fisher_cross_down:
                new_signal = -SHORT_SIZE
            elif is_trending and kama_bearish and fisher_cross_down and adx_4h[i] > 20:
                new_signal = -SHORT_SIZE
            elif kama_bearish and fisher[i] > 1.0 and rsi_14[i] > 60:
                new_signal = -SHORT_SIZE * 0.8
            elif bear_regime and rsi_overbought and close[i] < kama_4h[i]:
                new_signal = -SHORT_SIZE
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # Fisher reversal exit
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.5:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and bear_regime and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and kama_bullish:
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