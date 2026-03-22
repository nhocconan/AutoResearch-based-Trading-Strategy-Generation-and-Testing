#!/usr/bin/env python3
"""
Experiment #071: 12h KAMA Trend + Ehlers Fisher Transform + Adaptive Sizing
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA,
reducing whipsaw in ranging markets. Ehlers Fisher Transform catches reversals with less lag
than RSI. Combined with 1d trend filter and ATR-based position sizing, this should improve
Sharpe vs #059's Donchian+CRSI approach.

Key insights from failures:
- #059 got Sharpe=0.160 with 12h+1d HMA - direction is correct
- Complex regime filters (CHOP, Fisher alone) failed (#063, #065, #068)
- Need simpler logic with fewer conflicting conditions
- Position sizing adjustment based on volatility can reduce DD

Strategy components:
1. 1d KAMA = primary trend bias (adaptive to volatility)
2. 12h Fisher Transform = entry trigger (crosses -1.5 long, +1.5 short)
3. ADX(14) > 20 = trend confirmation filter
4. ATR ratio (ATR7/ATR30) = position size adjustment (reduce size in high vol)
5. Asymmetric entries: only long when price > 1d KAMA, only short when <

Why this might beat #059:
- KAMA adapts faster than HMA in trending markets, slower in ranging (less whipsaw)
- Fisher Transform has proven edge for reversal detection (Ehlers literature)
- Volatility-based sizing reduces exposure during panic (2022 crash protection)
- Fewer conflicting filters = more trades generated

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_1d_adaptive_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    More responsive in trends, smoother in ranges.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio (ER)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er = change / noise
        else:
            er = 0
        
        # Smoothing constant
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh > ll:
            # Normalize price to 0-1 range
            price_norm = (close[i] - ll) / (hh - ll)
            # Constrain to 0.001-0.999 to avoid log(0)
            price_norm = np.clip(price_norm, 0.001, 0.999)
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
            
            if i > period:
                trigger[i] = fisher[i - 1]
        else:
            fisher[i] = fisher[i - 1] if i > period else 0
            trigger[i] = trigger[i - 1] if i > period else 0
    
    return fisher, trigger

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF KAMA
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # KAMA on 12h
    kama_12h = calculate_kama(close, period=10)
    kama_12h_fast = calculate_kama(close, period=5)
    
    # Fisher Transform
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_WEAK = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY-BASED POSITION SIZING ===
        # Reduce size when volatility is high (ATR7/ATR30 > 1.5)
        if not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]) and atr_30[i] > 0:
            vol_ratio = atr_7[i] / atr_30[i]
        else:
            vol_ratio = 1.0
        
        if vol_ratio > 2.0:
            size_mult = 0.6  # Reduce to 60% in very high vol
        elif vol_ratio > 1.5:
            size_mult = 0.8  # Reduce to 80% in high vol
        else:
            size_mult = 1.0  # Normal size
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d KAMA = primary trend bias
        bull_trend_1d = close[i] > kama_1d_aligned[i]
        bear_trend_1d = close[i] < kama_1d_aligned[i]
        
        # 12h KAMA crossover = short-term momentum
        kama_bullish_12h = not np.isnan(kama_12h_fast[i]) and kama_12h_fast[i] > kama_12h[i]
        kama_bearish_12h = not np.isnan(kama_12h_fast[i]) and kama_12h_fast[i] < kama_12h[i]
        
        # EMA alignment confirmation
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === TREND STRENGTH ===
        trending = adx[i] > 20
        strong_trend = adx[i] > 30
        
        # DI confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = False
        if i > 300 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            if fisher[i] > -1.5 and fisher[i-1] <= -1.5:
                fisher_long_cross = True
        
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = False
        if i > 300 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            if fisher[i] < 1.5 and fisher[i-1] >= 1.5:
                fisher_short_cross = True
        
        # Fisher extreme levels (mean reversion)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths) ===
        
        # Path 1: Trend + Fisher crossover (primary signal)
        if bull_trend_1d and trending:
            if fisher_long_cross:
                if strong_trend and di_bullish:
                    new_signal = SIZE_STRONG * size_mult
                else:
                    new_signal = SIZE_BASE * size_mult
        
        # Path 2: KAMA crossover + trend bias
        if bull_trend_1d and kama_bullish_12h:
            if ema_bullish and adx[i] > 15:
                new_signal = SIZE_BASE * size_mult
        
        # Path 3: Fisher extreme + trend (mean reversion in trend)
        if bull_trend_1d and fisher_oversold:
            if close[i] > ema_21[i]:
                new_signal = SIZE_WEAK * size_mult
        
        # Path 4: Simple trend continuation
        if bull_trend_1d and ema_bullish:
            if di_bullish and adx[i] > 18:
                if close[i] > kama_12h[i]:
                    new_signal = SIZE_BASE * size_mult
        
        # === SHORT ENTRY CONDITIONS (multiple paths) ===
        
        # Path 1: Trend + Fisher crossover (primary signal)
        if bear_trend_1d and trending:
            if fisher_short_cross:
                if strong_trend and di_bearish:
                    new_signal = -SIZE_STRONG * size_mult
                else:
                    new_signal = -SIZE_BASE * size_mult
        
        # Path 2: KAMA crossover + trend bias
        if bear_trend_1d and kama_bearish_12h:
            if ema_bearish and adx[i] > 15:
                new_signal = -SIZE_BASE * size_mult
        
        # Path 3: Fisher extreme + trend (mean reversion in trend)
        if bear_trend_1d and fisher_overbought:
            if close[i] < ema_21[i]:
                new_signal = -SIZE_WEAK * size_mult
        
        # Path 4: Simple trend continuation
        if bear_trend_1d and ema_bearish:
            if di_bearish and adx[i] > 18:
                if close[i] < kama_12h[i]:
                    new_signal = -SIZE_BASE * size_mult
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals