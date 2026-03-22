#!/usr/bin/env python3
"""
Experiment #264: 1d KAMA Trend with 1w HMA Bias and ADX Filter

Hypothesis: After 263 experiments, clear patterns emerge:
1. RSI strategies consistently FAIL (negative Sharpe across 6+ experiments)
2. Mean reversion FAILS on BTC/ETH in bear markets
3. KAMA (Kaufman Adaptive) outperforms EMA/HMA in volatile regimes
4. 1d timeframe needs fewer but higher-quality signals

This strategy uses:
1. 1d KAMA(10) - adapts to volatility, less whipsaw than EMA in 2022 crash
2. 1w HMA(21) - stronger HTF bias than 1d (weekly trend is more reliable)
3. ADX(14) > 18 - only trade when trend has strength (lower threshold = more trades)
4. ATR-based position sizing - reduce size when vol is high
5. 2.5*ATR trailing stoploss - appropriate for 1d timeframe
6. Asymmetric entries - only long when 1w HMA bullish, only short when bearish

Why 1d might work:
- Fewer bars = less fee drag (critical after 263 experiments showed fee churn kills profits)
- 1w HMA bias is stronger filter than 1d HMA
- KAMA adapts to market regime automatically (fast in trends, slow in ranges)
- Lower ADX threshold (18 vs 25) ensures >=10 trades per symbol

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_trend_1w_hma_adx_atr_v1"
timeframe = "1d"
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

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - fast during trends, slow during ranges.
    This is critical for avoiding 2022-style whipsaws.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(efficiency_period, n):
        signal = np.abs(close[i] - close[i - efficiency_period])
        noise = np.sum(np.abs(np.diff(close[i - efficiency_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[efficiency_period] = close[efficiency_period]
    
    for i in range(efficiency_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength regardless of direction.
    ADX > 25 = strong trend, ADX < 20 = range/no trend.
    """
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        # True Range
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_s[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / tr_s[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx[:] = adx_series.values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30  # Base position size
    SIZE_REDUCED = 0.20  # Reduced size in high vol
    SIZE_MAX = 0.35  # Maximum position size
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1w HMA = very strong directional bias (hard filter)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 18 = trend has strength (lower threshold = more trades for 1d)
        trend_strong = adx[i] > 18
        
        # === KAMA TREND DIRECTION ===
        # KAMA slope indicates short-term trend direction
        kama_bullish = kama[i] > kama[i-5] if i >= 5 and not np.isnan(kama[i-5]) else False
        kama_bearish = kama[i] < kama[i-5] if i >= 5 and not np.isnan(kama[i-5]) else False
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 1w bias up + ADX strong + KAMA bullish
        # Looser conditions to ensure >=10 trades per symbol on 1d timeframe
        long_conditions = (
            bull_trend_1w and  # 1w HMA bias bullish
            trend_strong and  # ADX shows trend strength
            kama_bullish  # KAMA slope bullish
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_1w and  # 1w HMA bias bearish
            trend_strong and  # ADX shows trend strength
            kama_bearish  # KAMA slope bearish
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0  # 1w trend reversed against long
            if position_side < 0 and bull_trend_1w:
                new_signal = 0.0  # 1w trend reversed against short
        
        # === ADX WEAKNESS EXIT ===
        # Exit if trend strength disappears (ADX drops below 15)
        if in_position and adx[i] < 15:
            new_signal = 0.0  # Trend lost strength
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals