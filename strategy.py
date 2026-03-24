#!/usr/bin/env python3
"""
Experiment #950: 1h Primary + 4h/1d HTF — Regime-Adaptive CRSI + Choppiness

Hypothesis: Different market regimes require different strategies.
- Range markets (CHOP > 61.8): Mean reversion with CRSI extremes works best
- Trend markets (CHOP < 38.2): Follow HTF trend direction only

Key innovations:
1. Choppiness Index (CHOP) regime detection - distinguishes trend vs range
2. Connors RSI (CRSI) for precise mean reversion entries in range regime
3. 4h HMA(21) for intermediate trend bias
4. 1d HMA(50) for long-term trend confirmation
5. Session filter: 08-20 UTC (high liquidity hours)
6. Regime-adaptive: different entry logic per CHOP value
7. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- 2022 crash was trending down: CHOP < 38.2, follow 4h/1d bearish bias
- 2025 bear/range: CHOP > 61.8, CRSI mean reversion captures swings
- Session filter reduces false signals during low-liquidity hours
- Loose CRSI thresholds (15/85 not 10/90) ensure sufficient trades

Entry conditions (calibrated for 40-80 trades/year on 1h):
- LONG range: CHOP>61.8 + CRSI<15 + price>SMA200 + 4h HMA bull + session
- LONG trend: CHOP<38.2 + CRSI<40 + price>4h HMA + 1d HMA bull + session
- SHORT range: CHOP>61.8 + CRSI>85 + price<SMA200 + 4h HMA bear + session
- SHORT trend: CHOP<38.2 + CRSI>60 + price<4h HMA + 1d HMA bear + session

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_chop_htf_v1"
timeframe = "1h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = Range/Choppy market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if hh[i] > ll[i] and atr_sum[i] > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
    
    return choppiness

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Fast RSI for short-term momentum
    RSI(Streak): RSI of consecutive up/down days
    PercentRank: Where current close ranks vs last 100 closes
    
    CRSI < 10-15: Oversold (long opportunity)
    CRSI > 85-90: Overbought (short opportunity)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan, dtype=np.float64)
    for i in range(rsi_period, n):
        if avg_loss[i] == 0:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1.0
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1.0
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        if avg_streak_loss[i] == 0:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank(100)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
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
    
    hma = wma(diff, sqrt_n)
    return hma

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range_regime = choppiness[i] > 61.8
        is_trend_regime = choppiness[i] < 38.2
        
        # === HTF BIAS (4h HMA + 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === LONG TERM BIAS (SMA200) ===
        lt_bull = close[i] > sma_200[i]
        lt_bear = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with CRSI
        if is_range_regime and in_session:
            # LONG: CRSI oversold + above SMA200 + 4h bullish
            if crsi[i] < 15 and lt_bull and htf_4h_bull:
                desired_signal = SIZE_BASE
            # Strong long: very oversold + all bullish
            elif crsi[i] < 10 and lt_bull and htf_4h_bull and htf_1d_bull:
                desired_signal = SIZE_STRONG
            
            # SHORT: CRSI overbought + below SMA200 + 4h bearish
            elif crsi[i] > 85 and lt_bear and htf_4h_bear:
                desired_signal = -SIZE_BASE
            # Strong short: very overbought + all bearish
            elif crsi[i] > 90 and lt_bear and htf_4h_bear and htf_1d_bear:
                desired_signal = -SIZE_STRONG
        
        # TREND REGIME: Follow HTF trend with CRSI pullback entry
        elif is_trend_regime and in_session:
            # LONG: 4h/1d bullish + CRSI pullback (not extreme)
            if htf_4h_bull and htf_1d_bull and crsi[i] < 40:
                desired_signal = SIZE_BASE
            # Strong long: CRSI very low pullback
            elif htf_4h_bull and htf_1d_bull and crsi[i] < 30:
                desired_signal = SIZE_STRONG
            
            # SHORT: 4h/1d bearish + CRSI pullback (not extreme)
            elif htf_4h_bear and htf_1d_bear and crsi[i] > 60:
                desired_signal = -SIZE_BASE
            # Strong short: CRSI very high pullback
            elif htf_4h_bear and htf_1d_bear and crsi[i] > 70:
                desired_signal = -SIZE_STRONG
        
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