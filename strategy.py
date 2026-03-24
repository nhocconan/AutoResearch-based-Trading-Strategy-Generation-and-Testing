#!/usr/bin/env python3
"""
Experiment #012: 12h Primary + 1d HTF — KAMA Adaptive Trend + Connors RSI + Vol Regime

Hypothesis: After 11 failed experiments, the pattern is clear:
- Standard RSI fails in bear markets (gets stuck at extremes)
- Connor's RSI (CRSI) has proven 75% win rate for mean reversion
- KAMA adapts to volatility better than HMA/EMA (critical for 2022 crash)
- Volatility regime (ATR ratio) identifies panic/reversion opportunities
- Asymmetric logic: aggressive shorts in bear (ADX>25+price<SMA50), mean-revert in range
- This combines: KAMA trend (adaptive) + CRSI (proven) + Vol regime (panic detection)

Key design choices:
- Timeframe: 12h (20-50 trades/year, proven to work)
- HTF: 1d HMA for major trend bias
- Primary trend: KAMA(14) adaptive to volatility
- Entry: Connors RSI <20/>80 for mean reversion, KAMA breakout for trend
- Regime: ATR(7)/ATR(30) ratio >1.8 = vol spike (revert), <1.2 = calm (trend follow)
- Position size: 0.30 (30% of capital)
- Stoploss: 2.5x ATR trailing
- LOOSE filters to ensure >=30 trades on train, >=3 on test

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=30 train, >=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_crsi_volregime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency/volatility
    More responsive in trends, smoother in chop
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    Composite of: RSI(3) + RSI of streak length + Percentile rank
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period + rsi_period + streak_period:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_3 = np.zeros(n)
    rsi_3[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_3[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_3[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: Streak RSI
    streak = np.zeros(n)
    current_streak = 0
    last_direction = 0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            if last_direction > 0:
                current_streak += 1
            else:
                current_streak = 1
            last_direction = 1
        elif close[i] < close[i - 1]:
            if last_direction < 0:
                current_streak -= 1
            else:
                current_streak = -1
            last_direction = -1
        else:
            last_direction = 0
        streak[i] = current_streak
    
    # RSI of streak
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_streak_gain[i] / (avg_streak_loss[i] + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: Percentile Rank
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        rank = np.sum(window[:-1] < close[i]) / (rank_period - 1) * 100.0
        crsi[i] = (rsi_3[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

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
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate DM and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    # Smooth DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period*2:] = adx_raw[period*2:]
    
    return adx

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    
    # Volatility regime: ATR ratio
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    for i in range(n):
        if atr_30[i] > 1e-10 and not np.isnan(atr_7[i]):
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME ===
        # vol_ratio > 1.8 = panic/spike (mean revert)
        # vol_ratio < 1.2 = calm (trend follow)
        is_vol_spike = vol_ratio[i] > 1.8
        is_calm = vol_ratio[i] < 1.2
        
        # === TREND REGIME (ADX + SMA50) ===
        is_trending = adx[i] > 25.0
        is_bear_trend = is_trending and close[i] < sma_50[i]
        is_bull_trend = is_trending and close[i] > sma_50[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_vol_spike:
            # VOLATILITY SPIKE REGIME: Mean reversion (panic selling/buying)
            # LONG: CRSI extreme oversold + HTF not strongly bear
            if crsi_extreme_oversold and not (htf_bear and adx[i] > 30):
                desired_signal = SIZE
            # SHORT: CRSI extreme overbought + HTF not strongly bull
            elif crsi_extreme_overbought and not (htf_bull and adx[i] > 30):
                desired_signal = -SIZE
            # Fallback: regular CRSI extremes
            elif crsi_oversold and htf_bull:
                desired_signal = SIZE * 0.7
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE * 0.7
        
        elif is_bear_trend:
            # BEAR TREND REGIME: Only short retracements (asymmetric)
            # SHORT: CRSI overbought retracement + KAMA bear
            if crsi_overbought and kama_bear:
                desired_signal = -SIZE
            # LONG only on extreme oversold (counter-trend, smaller size)
            elif crsi_extreme_oversold:
                desired_signal = SIZE * 0.5
        
        elif is_bull_trend:
            # BULL TREND REGIME: Only long pullbacks (asymmetric)
            # LONG: CRSI oversold pullback + KAMA bull
            if crsi_oversold and kama_bull:
                desired_signal = SIZE
            # SHORT only on extreme overbought (counter-trend, smaller size)
            elif crsi_extreme_overbought:
                desired_signal = -SIZE * 0.5
        
        else:
            # RANGE/CHOP REGIME (ADX < 25): Mean reversion at extremes
            # LONG: CRSI oversold + HTF bull bias
            if crsi_oversold and htf_bull:
                desired_signal = SIZE
            # SHORT: CRSI overbought + HTF bear bias
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE
            # Fallback: KAMA breakout in calm conditions
            elif is_calm and kama_bull and htf_bull:
                desired_signal = SIZE * 0.7
            elif is_calm and kama_bear and htf_bear:
                desired_signal = -SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals