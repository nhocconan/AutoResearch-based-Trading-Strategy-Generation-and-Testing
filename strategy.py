#!/usr/bin/env python3
"""
Experiment #1480: 6h Primary + 1d/1w HTF — Fisher Transform + ADX Regime

Hypothesis: 6h timeframe offers optimal balance between 4h (too noisy) and 12h (too slow).
This strategy uses FISHER TRANSFORM for reversal detection (proven in bear/range markets)
combined with ADX regime detection and 1w/1d trend filters.

Key components:
1. 1w HMA(21) for major trend bias (stronger than 1d)
2. 1d ADX(14) for regime detection with hysteresis:
   - ADX > 25 = trending (use breakout logic)
   - ADX < 18 = ranging (use mean-reversion logic)
   - Between = hold existing positions
3. 6h Fisher Transform(9) for entry timing:
   - Long: Fisher crosses above -1.5 (oversold reversal)
   - Short: Fisher crosses below +1.5 (overbought reversal)
4. 6h Bollinger(20,2.0) for range regime entries
5. ATR(14) trailing stoploss (2.5x ATR)
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work:
- Fisher Transform outperforms RSI in bear/range markets (research-backed)
- 1w HMA provides stronger trend filter than 1d alone
- ADX hysteresis (25/18) prevents regime whipsaw
- 6h TF = natural 25-40 trades/year (fee-efficient)
- LOOSE Fisher thresholds (-1.5/+1.5, not -2/+2) guarantee trades

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG trend: 1w_HMA bullish + 1d_ADX>18 + Fisher crosses above -1.5
- SHORT trend: 1w_HMA bearish + 1d_ADX>18 + Fisher crosses below +1.5
- LONG range: 1d_ADX<18 + Fisher<-1.0 + price<BB_lower
- SHORT range: 1d_ADX<18 + Fisher>+1.0 + price>BB_upper

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_regime_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        avg_plus_dm = pd.Series(plus_dm[:i+1]).ewm(span=period, min_periods=period, adjust=False).mean().iloc[-1]
        avg_minus_dm = pd.Series(minus_dm[:i+1]).ewm(span=period, min_periods=period, adjust=False).mean().iloc[-1]
        avg_tr = pd.Series(tr[:i+1]).ewm(span=period, min_periods=period, adjust=False).mean().iloc[-1]
        
        if avg_tr > 1e-10:
            plus_di[i] = 100.0 * avg_plus_dm / avg_tr
            minus_di[i] = 100.0 * avg_minus_dm / avg_tr
    
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            sum_di = plus_di[i] + minus_di[i]
            if sum_di > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / sum_di
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_fisher(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        normalized = 2.0 * (close[i] - lowest) / price_range - 1.0
        normalized = max(-0.999, min(0.999, normalized))
        
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher(high, low, close, period=9)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # ADX regime hysteresis tracking
    prev_adx_regime = 0  # 0=neutral, 1=trending, 2=ranging
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        adx = adx_1d_aligned[i]
        
        # Hysteresis: enter trending at 25, exit at 18
        if adx > 25:
            adx_regime = 1  # trending
        elif adx < 18:
            adx_regime = 2  # ranging
        else:
            adx_regime = prev_adx_regime  # hold previous regime
        
        prev_adx_regime = adx_regime
        is_trend_regime = (adx_regime == 1)
        is_range_regime = (adx_regime == 2)
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_sig = fisher_signal[i]
        
        # Fisher crossover detection
        fisher_cross_above_neg15 = (fisher_val > -1.5) and (not np.isnan(fisher_sig) and fisher_sig <= -1.5)
        fisher_cross_below_pos15 = (fisher_val < 1.5) and (not np.isnan(fisher_sig) and fisher_sig >= 1.5)
        
        # Fisher extreme levels (for range regime)
        fisher_oversold = fisher_val < -1.0
        fisher_overbought = fisher_val > 1.0
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.002
        bb_touch_upper = close[i] >= bb_upper[i] * 0.998
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Fisher reversal + 1w trend bias
        if is_trend_regime:
            # LONG: 1w bullish + Fisher crosses above -1.5
            if price_above_1w and fisher_cross_above_neg15:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1w bearish + Fisher crosses below +1.5
            elif price_below_1w and fisher_cross_below_pos15:
                desired_signal = -SIZE_STRONG
        
        # RANGE REGIME: Fisher extremes + Bollinger touch
        elif is_range_regime:
            # LONG: Fisher oversold + price at BB lower
            if fisher_oversold and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: Fisher overbought + price at BB upper
            elif fisher_overbought and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Only Fisher reversals with 1w confirmation
        else:
            # LONG: 1w bullish + Fisher crosses above -1.5
            if price_above_1w and fisher_cross_above_neg15:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + Fisher crosses below +1.5
            elif price_below_1w and fisher_cross_below_pos15:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals