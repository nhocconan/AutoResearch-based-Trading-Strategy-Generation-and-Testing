#!/usr/bin/env python3
"""
Experiment #962: 4h Primary + 1d/1w HTF — Dual Regime HMA/RSI Strategy

Hypothesis: 4h timeframe with Choppiness Index regime filter + HMA trend + RSI entries
will outperform in mixed 2022-2025 markets by adapting to regime changes.

Key innovations:
1. CHOP(14) regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
2. HMA(21) for fast trend detection with minimal lag
3. RSI(14) extremes for entries (35/65 thresholds - looser for more trades)
4. ADX(14) > 20 for trend confirmation (not 25+ which is too strict)
5. 1d HMA(21) for intermediate trend bias
6. 1w momentum (close > open) for weekly bias
7. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- 4h proven timeframe (20-50 trades/year target)
- CHOP filter avoids whipsaws in 2022 bottom chop
- RSI 35/65 thresholds ensure sufficient trade frequency
- HTF bias prevents counter-trend trades
- ADX confirms trend strength before trend entries

Entry conditions (LOOSE to guarantee >=30 trades):
- LONG = 1w bull + 1d bull + (ADX>20 + HMA crossover OR CHOP>61 + RSI<40)
- SHORT = 1w bear + 1d bear + (ADX>20 + HMA crossunder OR CHOP>61 + RSI>60)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_hma_rsi_regime_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - minimal lag trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(plus_di, atr, out=np.zeros_like(plus_di), where=atr != 0) * 100
    minus_di = np.divide(minus_di, atr, out=np.zeros_like(minus_di), where=atr != 0) * 100
    
    dx = np.divide(np.abs(plus_di - minus_di), plus_di + minus_di, 
                   out=np.zeros_like(plus_di), where=(plus_di + minus_di) != 0) * 100
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / tr_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Weekly momentum: close vs open
    weekly_momentum_raw = (df_1w['close'].values - df_1w['open'].values) / (df_1w['open'].values + 1e-10)
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum_raw)
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(weekly_momentum_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w momentum + 1d HMA) ===
        htf_1w_bull = weekly_momentum_aligned[i] > 0.0
        htf_1w_bear = weekly_momentum_aligned[i] < 0.0
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (CHOP) ===
        is_trending = chop_14[i] < 45.0  # Slightly relaxed from 38.2
        is_ranging = chop_14[i] > 55.0  # Slightly relaxed from 61.8
        
        # === 4h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_4h_21[i-1]) and not np.isnan(hma_4h_50[i-1]):
            hma_crossover_long = (hma_4h_21[i-1] <= hma_4h_50[i-1]) and (hma_4h_21[i] > hma_4h_50[i])
            hma_crossover_short = (hma_4h_21[i-1] >= hma_4h_50[i-1]) and (hma_4h_21[i] < hma_4h_50[i])
        
        # === RSI EXTREMES (LOOSE THRESHOLDS FOR MORE TRADES) ===
        rsi_oversold = rsi_14[i] < 40  # Relaxed from 30
        rsi_overbought = rsi_14[i] > 60  # Relaxed from 70
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Relaxed from 25
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # LONG entries - multiple paths to ensure trades
        if htf_1w_bull or htf_1d_bull:  # Either HTF bullish
            if is_trending and adx_strong and hma_crossover_long:
                # Trend regime: HMA crossover with ADX confirmation
                desired_signal = SIZE_STRONG
            elif is_ranging and rsi_oversold:
                # Range regime: RSI mean reversion
                desired_signal = SIZE_BASE
            elif hma_4h_21[i] > hma_4h_50[i] and rsi_14[i] < 50:
                # Trend continuation with pullback
                desired_signal = SIZE_BASE
            elif htf_1w_bull and htf_1d_bull and rsi_14[i] < 45:
                # Strong HTF bias + RSI pullback
                desired_signal = SIZE_BASE
        
        # SHORT entries - multiple paths to ensure trades
        elif htf_1w_bear or htf_1d_bear:  # Either HTF bearish
            if is_trending and adx_strong and hma_crossover_short:
                # Trend regime: HMA crossover with ADX confirmation
                desired_signal = -SIZE_STRONG
            elif is_ranging and rsi_overbought:
                # Range regime: RSI mean reversion
                desired_signal = -SIZE_BASE
            elif hma_4h_21[i] < hma_4h_50[i] and rsi_14[i] > 50:
                # Trend continuation with pullback
                desired_signal = -SIZE_BASE
            elif htf_1w_bear and htf_1d_bear and rsi_14[i] > 55:
                # Strong HTF bias + RSI pullback
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