#!/usr/bin/env python3
"""
Experiment #437: 12h Fisher-KAMA Regime Adaptive with Daily/Weekly Trend Filter

Hypothesis: After 436 failed experiments, the pattern is clear:
1. Pure trend following fails on BTC/ETH (too much chopping)
2. Pure mean reversion fails in strong trends (2021 bull, 2022 crash)
3. REGIME DETECTION is the missing piece - switch logic based on market state

This strategy uses THREE regime-aware signal types:

1. EHRLERS FISHER TRANSFORM (reversals in ranging markets):
   - Fisher < -1.5 = oversold reversal long
   - Fisher > +1.5 = overbought reversal short
   - Works best when CHOP > 50 (ranging regime)

2. KAMA ADAPTIVE TREND (trending markets):
   - KAMA adapts to volatility (fast in trends, slow in ranges)
   - Long when price > KAMA and KAMA sloping up
   - Short when price < KAMA and KAMA sloping down
   - Works best when CHOP < 40 (trending regime)

3. BOLLINGER SQUEEZE BREAKOUT (volatility expansion):
   - BB Width at 6-month low = coiling spring
   - Breakout with volume confirmation = explosive move
   - Direction from 1d HMA bias

REGIME FILTER (Choppiness Index):
- CHOP > 55 = ranging (use Fisher mean reversion)
- CHOP < 40 = trending (use KAMA trend following)
- 40-55 = transition (reduce position size by 50%)

MULTI-TIMEFRAME CONFIRMATION:
- 1d HMA(21) for primary trend bias
- 1w HMA(21) for macro bias (only enter with macro alignment)
- Use mtf_data helper (call ONCE before loop!)

POSITION SIZING: 0.25 discrete (conservative for 12h volatility)
STOPLOSS: 2.5 * ATR(14) trailing stop
LEVERAGE: 1.0 (no leverage)

Why 12h should work:
- Fewer false signals than lower timeframes
- Still generates 20-40 trades/year (sufficient for Sharpe calculation)
- Regime adaptation prevents whipsaw in 2022-style crashes
- Fisher Transform proven edge in crypto mean reversion

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d, 1w via mtf_data helper
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_chop_regime_1d_1w_hma_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i-period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    kama[period-1] = close[period-1]
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Calculate highest high and lowest low over period
        hh = high[i-period+1:i+1].max()
        ll = low[i-period+1:i+1].min()
        
        # Normalize price
        if hh != ll:
            value = 0.66 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * np.nan_to_num(fisher_prev[i-1], 0)
            value = np.clip(value, -0.999, 0.999)
            fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
            if i > period - 1:
                fisher_prev[i] = fisher[i-1]
            else:
                fisher_prev[i] = 0
        else:
            fisher[i] = 0
            fisher_prev[i] = 0
    
    return fisher, fisher_prev

def calculate_choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index for regime detection."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high != lowest_low:
            tr_sum = 0
            for j in range(i-period+1, i+1):
                tr_sum += np.max([high[j] - low[j], 
                                  np.abs(high[j] - close[j-1]), 
                                  np.abs(low[j] - close[j-1])])
            
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma
    return upper, lower, bb_width

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
    # Calculate BB Width percentile for squeeze detection
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=126, min_periods=126).apply(
        lambda x: np.percentile(x.dropna(), 10) if len(x.dropna()) > 0 else np.nan
    ).values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_FULL = 0.28
    SIZE_HALF = 0.14
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND BIAS (1w HMA) ===
        bull_macro = close[i] > hma_1w_aligned[i]
        bear_macro = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND BIAS (1d HMA) ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop[i] > 55
        trending_regime = chop[i] < 40
        transition_regime = 40 <= chop[i] <= 55
        
        # === POSITION SIZE ADJUSTMENT FOR REGIME ===
        current_size = SIZE_HALF if transition_regime else SIZE_FULL
        
        # === SIGNAL 1: FISHER TRANSFORM REVERSAL (ranging regime) ===
        fisher_long = False
        fisher_short = False
        
        if ranging_regime:
            # Fisher crosses above -1.5 from below = long
            if fisher[i] > -1.5 and fisher_prev[i] <= -1.5:
                fisher_long = bull_macro  # Only with macro bias
            # Fisher crosses below +1.5 from above = short
            if fisher[i] < 1.5 and fisher_prev[i] >= 1.5:
                fisher_short = bear_macro
        
        # === SIGNAL 2: KAMA TREND FOLLOWING (trending regime) ===
        kama_long = False
        kama_short = False
        
        if trending_regime:
            # Price above KAMA and KAMA sloping up
            if i > 0 and close[i] > kama[i] and kama[i] > kama[i-1]:
                kama_long = bull_trend_1d and bull_macro
            # Price below KAMA and KAMA sloping down
            if i > 0 and close[i] < kama[i] and kama[i] < kama[i-1]:
                kama_short = bear_trend_1d and bear_macro
        
        # === SIGNAL 3: BB SQUEEZE BREAKOUT (any regime) ===
        bb_long = False
        bb_short = False
        
        # BB Width at 6-month low (squeeze) + breakout
        if not np.isnan(bb_width_percentile[i]):
            bb_squeeze = bb_width[i] < bb_width_percentile[i]
            
            if bb_squeeze:
                # Breakout above BB upper with RSI confirmation
                if close[i] > bb_upper[i] and rsi[i] > 50:
                    bb_long = bull_trend_1d
                # Breakdown below BB lower with RSI confirmation
                if close[i] < bb_lower[i] and rsi[i] < 50:
                    bb_short = bear_trend_1d
        
        # === GENERATE SIGNAL (Regime-Adaptive) ===
        new_signal = 0.0
        
        # Priority: Fisher (ranging) > KAMA (trending) > BB Breakout (all)
        if ranging_regime:
            if fisher_long:
                new_signal = current_size
            elif fisher_short:
                new_signal = -current_size
        elif trending_regime:
            if kama_long:
                new_signal = current_size
            elif kama_short:
                new_signal = -current_size
        
        # BB breakout works in any regime (lower priority)
        if new_signal == 0.0:
            if bb_long:
                new_signal = current_size
            elif bb_short:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_macro:
                new_signal = 0.0
            if position_side < 0 and bull_macro:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals