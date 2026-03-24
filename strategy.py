#!/usr/bin/env python3
"""
Experiment #866: 1d Primary + 1w HTF — Regime Adaptive Trend/Mean-Revert

Hypothesis: Daily timeframe with weekly HTF bias provides optimal balance of
signal quality and trade frequency (20-40 trades/year). Using Choppiness Index
to switch between trend-following (CHOP<50) and mean-reversion (CHOP>50) modes.
Weekly HMA(21) provides strong HTF bias filter. Loose entry conditions ensure
sufficient trades while ATR trailing stop controls drawdown.

Key innovations:
1. 1w HMA(21) for HTF trend bias - very smooth, lag acceptable on 1d
2. 1d HMA(16/48) dual crossover for local trend confirmation
3. RSI(14) extremes for mean-reversion entries (loose: <35/>65)
4. Choppiness Index(14) regime switch: >50 = range, <50 = trend
5. ADX(14) confirmation for trend strength (>20 = trending)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- TREND REGIME (CHOP<50 + ADX>20): LONG = 1w HMA bull + 1d HMA crossover up
- TREND REGIME (CHOP<50 + ADX>20): SHORT = 1w HMA bear + 1d HMA crossover down
- RANGE REGIME (CHOP>50): LONG = 1w HMA bull + RSI<35
- RANGE REGIME (CHOP>50): SHORT = 1w HMA bear + RSI>65

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_chop_adx_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 50 as threshold for regime switch
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/range
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # DX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
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
        
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_1d_16[i-1]) and not np.isnan(hma_1d_48[i-1]):
            hma_crossover_long = (hma_1d_16[i-1] <= hma_1d_48[i-1]) and (hma_1d_16[i] > hma_1d_48[i])
            hma_crossover_short = (hma_1d_16[i-1] >= hma_1d_48[i-1]) and (hma_1d_16[i] < hma_1d_48[i])
        
        # === HMA TREND ===
        hma_1d_bull = hma_1d_16[i] > hma_1d_48[i]
        hma_1d_bear = hma_1d_16[i] < hma_1d_48[i]
        
        # === RSI CONDITIONS (LOOSE for trades) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0
        chop_ranging = chop_14[i] >= 50.0
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20.0
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if htf_1w_bull:
            # Bullish HTF bias - prefer longs
            if chop_trending and adx_strong:
                # Trend regime: use HMA crossover or trend
                if hma_crossover_long:
                    desired_signal = SIZE_STRONG
                elif hma_1d_bull:
                    desired_signal = SIZE_BASE
            else:
                # Range regime: use RSI mean reversion
                if rsi_oversold:
                    desired_signal = SIZE_STRONG
                elif rsi_14[i] < 45.0:
                    desired_signal = SIZE_BASE
        
        elif htf_1w_bear:
            # Bearish HTF bias - prefer shorts
            if chop_trending and adx_strong:
                # Trend regime: use HMA crossover or trend
                if hma_crossover_short:
                    desired_signal = -SIZE_STRONG
                elif hma_1d_bear:
                    desired_signal = -SIZE_BASE
            else:
                # Range regime: use RSI mean reversion
                if rsi_overbought:
                    desired_signal = -SIZE_STRONG
                elif rsi_14[i] > 55.0:
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