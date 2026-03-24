#!/usr/bin/env python3
"""
Experiment #982: 4h Primary + 1d/1w HTF — Dual Regime with ADX Confirmation

Hypothesis: 4h timeframe with adaptive regime switching (trend vs mean-revert)
based on Choppiness Index + ADX confirmation will outperform single-regime strategies.

Key innovations:
1. CHOP(14) + ADX(14) dual regime filter:
   - CHOP < 38.2 AND ADX > 25 = strong trend (trend follow)
   - CHOP > 61.8 AND ADX < 20 = strong range (mean revert)
   - Otherwise = neutral (reduce position or wait)
2. 1d HMA(21) for intermediate trend bias
3. 1w momentum (close > open) for weekly directional bias
4. Trend entries: HMA(8/21) crossover + RSI(14) pullback to 40-60
5. Range entries: RSI(14) extremes (25/75) + Bollinger Band touch
6. ATR(14) 2.5x trailing stop for risk management
7. LOOSE entry thresholds to guarantee 30+ trades/year

Why this should work:
- Dual filter (CHOP + ADX) reduces false regime signals
- 4h captures multi-day swings without 1h noise or 6h lag
- HTF bias prevents counter-trend trades in strong moves
- Relaxed RSI thresholds (25/75 instead of 30/70) for more trades
- Discrete sizing (0.25/0.30) minimizes fee churn

Entry conditions (LOOSE to guarantee trades):
- LONG trend = 1w bull + 1d bull + CHOP<38 + ADX>25 + HMA crossover + RSI>40
- LONG range = 1d bull + CHOP>61 + ADX<20 + RSI<30 + price<BBLow
- SHORT trend = 1w bear + 1d bear + CHOP<38 + ADX>25 + HMA crossunder + RSI<60
- SHORT range = 1d bear + CHOP>61 + ADX<20 + RSI>70 + price>BBHigh

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_adx_chop_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        plus_smooth = np.sum(plus_dm[i-period+1:i+1])
        minus_smooth = np.sum(minus_dm[i-period+1:i+1])
        tr_smooth = np.sum(tr[i-period+1:i+1])
        
        if tr_smooth > 1e-10:
            plus_di[i] = 100.0 * plus_smooth / tr_smooth
            minus_di[i] = 100.0 * minus_smooth / tr_smooth
    
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index"""
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

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
    hma_4h_8 = calculate_hma(close, period=8)
    hma_4h_21 = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_8[i]) or np.isnan(hma_4h_21[i]):
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
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
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
        
        # === REGIME DETECTION (CHOP + ADX) ===
        is_strong_trend = (chop_14[i] < 38.2) and (adx_14[i] > 25)
        is_strong_range = (chop_14[i] > 61.8) and (adx_14[i] < 20)
        
        # === 4h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_4h_8[i-1]) and not np.isnan(hma_4h_21[i-1]):
            hma_crossover_long = (hma_4h_8[i-1] <= hma_4h_21[i-1]) and (hma_4h_8[i] > hma_4h_21[i])
            hma_crossover_short = (hma_4h_8[i-1] >= hma_4h_21[i-1]) and (hma_4h_8[i] < hma_4h_21[i])
        
        # === RSI CONDITIONS (LOOSE THRESHOLDS FOR MORE TRADES) ===
        rsi_oversold = rsi_14[i] < 30  # Relaxed from 25
        rsi_overbought = rsi_14[i] > 70  # Relaxed from 75
        rsi_pullback_long = 40 < rsi_14[i] < 55
        rsi_pullback_short = 45 < rsi_14[i] < 60
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_low = close[i] <= bb_lower[i] * 1.001
        bb_touch_high = close[i] >= bb_upper[i] * 0.999
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        desired_signal = 0.0
        
        # LONG entries - TREND REGIME
        if is_strong_trend and htf_1w_bull and htf_1d_bull:
            if hma_crossover_long and rsi_14[i] > 40:
                desired_signal = SIZE_STRONG
            elif hma_4h_8[i] > hma_4h_21[i] and rsi_pullback_long:
                desired_signal = SIZE_BASE
        
        # LONG entries - RANGE REGIME
        elif is_strong_range and htf_1d_bull:
            if rsi_oversold and bb_touch_low:
                desired_signal = SIZE_BASE
            elif rsi_14[i] < 35 and close[i] < bb_mid[i]:
                desired_signal = SIZE_BASE
        
        # SHORT entries - TREND REGIME
        elif is_strong_trend and htf_1w_bear and htf_1d_bear:
            if hma_crossover_short and rsi_14[i] < 60:
                desired_signal = -SIZE_STRONG
            elif hma_4h_8[i] < hma_4h_21[i] and rsi_pullback_short:
                desired_signal = -SIZE_BASE
        
        # SHORT entries - RANGE REGIME
        elif is_strong_range and htf_1d_bear:
            if rsi_overbought and bb_touch_high:
                desired_signal = -SIZE_BASE
            elif rsi_14[i] > 65 and close[i] > bb_mid[i]:
                desired_signal = -SIZE_BASE
        
        # === CONTINUATION SIGNALS (add more trades) ===
        # If already in strong trend, allow continuation entries
        if htf_1w_bull and htf_1d_bull and hma_4h_8[i] > hma_4h_21[i]:
            if rsi_14[i] < 50 and desired_signal == 0:
                desired_signal = SIZE_BASE * 0.5
        
        if htf_1w_bear and htf_1d_bear and hma_4h_8[i] < hma_4h_21[i]:
            if rsi_14[i] > 50 and desired_signal == 0:
                desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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