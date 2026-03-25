#!/usr/bin/env python3
"""
Experiment #1623: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend with ADX Regime

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). KAMA (Kaufman Adaptive 
Moving Average) adapts to market volatility - fast in trends, slow in chop. Combined with 
ADX for trend strength and dual HTF (1w direction + 1d bias), this captures regime changes 
better than static EMAs/HMAs which failed repeatedly.

Key design choices based on failure analysis:
1. KAMA instead of HMA/EMA - adapts ER (Efficiency Ratio) to market conditions
2. ADX(14) for trend strength - ADX>25 = trend, ADX<20 = range (hysteresis built in)
3. Triple HTF confirmation: 1w HMA direction + 1d HMA bias + 6h KAMA entry
4. LOOSE entry thresholds to guarantee ≥30 trades/train (learned from #1613,#1616,#1617,#1619)
5. Asymmetric sizing: 0.30 when all HTF align, 0.25 when partial alignment
6. 2.5x ATR trailing stoploss via signal→0

Why 6h might work where 4h/12h struggled:
- 6h = 4 bars/day vs 4h = 6 bars/day → fewer trades, less fee drag
- 6h = 28 bars/week vs 12h = 14 bars/week → enough data for statistical edge
- Catches multi-day trends without 12h lag

Entry logic (LOOSE to guarantee trades):
- TREND (ADX>25): KAMA cross + 1w/1d HMA alignment + ADX confirmation
- RANGE (ADX<20): KAMA mean reversion + 1d HMA bias only
- NEUTRAL (20<ADX<25): 1d HMA bias + KAMA slope

Target: Sharpe>0.6, trades≥30 train, trades≥3 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_adx_regime_1w1d_loose_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    ER = |net change| / sum of absolute changes
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio calculation
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    # Initialize with SMA
    kama[period + slow_period - 1] = np.nanmean(close[period:period + slow_period])
    
    for i in range(period + slow_period, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    mask = tr_s > 1e-10
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    # DX
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period * 2, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

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
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    adx_14 = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
    
    # Warmup period
    min_bars = 60
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX) ===
        adx = adx_14[i]
        is_trend_regime = adx > 25.0
        is_range_regime = adx < 20.0
        
        # === TREND DIRECTION (HTF HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA SIGNALS ===
        kama_val = kama_10[i]
        kama_prev = kama_10[i-1] if i > 0 and not np.isnan(kama_10[i-1]) else kama_val
        
        # KAMA cross above/below price
        kama_cross_above = close[i] > kama_val and (i > 0 and close[i-1] <= kama_prev if not np.isnan(kama_prev) else False)
        kama_cross_below = close[i] < kama_val and (i > 0 and close[i-1] >= kama_prev if not np.isnan(kama_prev) else False)
        
        # KAMA slope
        kama_slope_up = kama_val > kama_prev if not np.isnan(kama_prev) else False
        kama_slope_down = kama_val < kama_prev if not np.isnan(kama_prev) else False
        
        # === RSI CONFIRMATION (LOOSE) ===
        rsi_val = rsi_14[i]
        rsi_bullish = rsi_val > 40
        rsi_bearish = rsi_val < 60
        rsi_oversold = rsi_val < 45
        rsi_overbought = rsi_val > 55
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME (ADX > 25): KAMA + HTF alignment
        if is_trend_regime:
            # LONG: 1w bullish + 1d bullish + KAMA cross above + RSI bullish
            if price_above_1w and price_above_1d and kama_cross_above and rsi_bullish:
                desired_signal = SIZE_STRONG  # All HTF align
            elif price_above_1d and kama_cross_above and rsi_bullish:
                desired_signal = SIZE_BASE  # Partial alignment
            
            # SHORT: 1w bearish + 1d bearish + KAMA cross below + RSI bearish
            elif price_below_1w and price_below_1d and kama_cross_below and rsi_bearish:
                desired_signal = -SIZE_STRONG  # All HTF align
            elif price_below_1d and kama_cross_below and rsi_bearish:
                desired_signal = -SIZE_BASE  # Partial alignment
        
        # RANGE REGIME (ADX < 20): Mean reversion with 1d bias only
        elif is_range_regime:
            # LONG: 1d bullish + RSI oversold + KAMA slope up
            if price_above_1d and rsi_oversold and kama_slope_up:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + RSI overbought + KAMA slope down
            elif price_below_1d and rsi_overbought and kama_slope_down:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME (20 <= ADX <= 25): Simple 1d bias + KAMA slope
        else:
            # LONG: 1d bullish + KAMA slope up + RSI not overbought
            if price_above_1d and kama_slope_up and rsi_val < 60:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + KAMA slope down + RSI not oversold
            elif price_below_1d and kama_slope_down and rsi_val > 40:
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