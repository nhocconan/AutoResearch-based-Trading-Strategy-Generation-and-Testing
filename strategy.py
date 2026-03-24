#!/usr/bin/env python3
"""
Experiment #883: 6h Primary + 1d/1w HTF — Keltner/Fisher/ADX Regime Adaptive

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow).
Using Keltner Channels for volatility-based entries (different from BB),
Ehlers Fisher Transform for reversal signals (proven in bear/range markets),
and triple HTF structure (1w bias + 1d trend + 6h entry) provides unique edge.

Key innovations:
1. 1w HMA(21) for ultra-long-term market bias (bull/bear regime)
2. 1d HMA(21) for intermediate trend direction
3. 6h Keltner Channels (EMA20 + 1.5*ATR) for volatility-based entry zones
4. Ehlers Fisher Transform(9) for reversal signals at extremes
5. ADX(14) for trend strength confirmation (threshold=20, loose)
6. Regime-adaptive: mean-revert when ADX<20, trend-follow when ADX>=20
7. ATR(14) 2.5x trailing stop for risk management

Entry conditions (LOOSE to ensure trades):
- LONG BIAS (1w HMA bull + 1d HMA bull):
  - ADX<20 (range): Fisher<-1.2 + price<Keltner_lower
  - ADX>=20 (trend): Price>Keltner_mid + Fisher crossing up from <-1.0
- SHORT BIAS (1w HMA bear + 1d HMA bear):
  - ADX<20 (range): Fisher>1.2 + price>Keltner_upper
  - ADX>=20 (trend): Price<Keltner_mid + Fisher crossing down from >1.0

Target: Sharpe>0.45, trades>=40 train, trades>=10 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_fisher_adx_triple_htf_v1"
timeframe = "6h"
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
    
    def wma(series, span):
        if span < 1:
            span = 1
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/range
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into a Gaussian distribution for clearer reversal signals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    Using looser thresholds (-1.2/+1.2) for more trades
    """
    n = len(close := (high + low) / 2)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        x = (close[i] - lowest) / price_range
        x = max(0.001, min(0.999, x))
        
        fisher_val = 0.5 * np.log((1 + x) / (1 - x))
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
        
        fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1] if i > period else fisher_val
    
    return fisher, fisher_prev

def calculate_keltner(high, low, close, ema_period=20, atr_period=14, multiplier=1.5):
    """
    Keltner Channels
    Middle = EMA(20)
    Upper = EMA(20) + multiplier * ATR(14)
    Lower = EMA(20) - multiplier * ATR(14)
    """
    n = len(close)
    ema_mid = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(n):
        if not np.isnan(ema_mid[i]) and not np.isnan(atr[i]):
            upper[i] = ema_mid[i] + multiplier * atr[i]
            lower[i] = ema_mid[i] - multiplier * atr[i]
    
    return ema_mid, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    keltner_mid, keltner_upper, keltner_lower = calculate_keltner(high, low, close, ema_period=20, atr_period=14, multiplier=1.5)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    adx_14 = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(keltner_mid[i]) or np.isnan(fisher[i]) or np.isnan(adx_14[i]):
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
        
        # === HTF BIAS (1w + 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Combined bias (both must agree for strong signal)
        strong_bull_bias = htf_1w_bull and htf_1d_bull
        strong_bear_bias = htf_1w_bear and htf_1d_bear
        
        # Weak bias (only 1d agrees, allows more trades)
        weak_bull_bias = htf_1d_bull
        weak_bear_bias = htf_1d_bear
        
        # === ADX REGIME ===
        adx_trending = adx_14[i] >= 20.0  # Loose threshold for more trades
        adx_ranging = adx_14[i] < 20.0
        
        # === FISHER SIGNALS ===
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        fisher_cross_up = (fisher_prev[i] < -1.0) and (fisher[i] >= -1.0)
        fisher_cross_down = (fisher_prev[i] > 1.0) and (fisher[i] <= 1.0)
        
        # === KELTNER POSITION ===
        price_below_lower = close[i] < keltner_lower[i]
        price_above_upper = close[i] > keltner_upper[i]
        price_above_mid = close[i] > keltner_mid[i]
        price_below_mid = close[i] < keltner_mid[i]
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if strong_bull_bias or weak_bull_bias:
            # Bullish bias - look for longs
            if adx_ranging:
                # Range regime: mean reversion at Keltner lower + Fisher oversold
                if price_below_lower and fisher_oversold:
                    desired_signal = SIZE_STRONG
                elif fisher_oversold:
                    desired_signal = SIZE_BASE
            else:
                # Trend regime: pullback to mid + Fisher turning up
                if price_above_mid and fisher_cross_up:
                    desired_signal = SIZE_STRONG
                elif price_above_mid and fisher[i] > -0.5:
                    desired_signal = SIZE_BASE
        
        elif strong_bear_bias or weak_bear_bias:
            # Bearish bias - look for shorts
            if adx_ranging:
                # Range regime: mean reversion at Keltner upper + Fisher overbought
                if price_above_upper and fisher_overbought:
                    desired_signal = -SIZE_STRONG
                elif fisher_overbought:
                    desired_signal = -SIZE_BASE
            else:
                # Trend regime: pullback to mid + Fisher turning down
                if price_below_mid and fisher_cross_down:
                    desired_signal = -SIZE_STRONG
                elif price_below_mid and fisher[i] < 0.5:
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