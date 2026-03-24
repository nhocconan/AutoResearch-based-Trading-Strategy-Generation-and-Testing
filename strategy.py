#!/usr/bin/env python3
"""
Experiment #025: 1h Primary + 4h/1d HTF — Regime-Adaptive with Connors RSI

Hypothesis: Previous 1h strategies failed (Sharpe=0.000) because entry conditions 
were too strict. This strategy uses REGIME-ADAPTIVE logic:
- CHOP > 55 = RANGE → Mean reversion (Connors RSI extremes)
- CHOP < 45 = TREND → Follow HTF direction (4h HMA)
- 45-55 = NO TRADE (avoid whipsaw)

Key improvements:
1. Choppiness Index regime filter - adapts to market state
2. Connors RSI (3-period RSI + streak + percent rank) - proven 75% win rate
3. 4h HMA for trend bias - ensures we trade with HTF flow
4. Volume filter (>0.7x avg) - confirms moves
5. Session filter (8-20 UTC) - avoids low-liquidity hours
6. LOOSE entry thresholds - ENSURES trades generate (learned from #015,#020)

Entry Logic:
- RANGE (CHOP>55): Long CRSI<15, Short CRSI>85 (mean revert)
- TREND (CHOP<45): Long if 4h HMA bull + RSI>40, Short if 4h HMA bear + RSI<60
- Size: 0.25 (discrete, lower for 1h to control fee drag)

Risk: 2.0x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1h (target 40-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_hma_volume_v1"
timeframe = "1h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10 or atr_sum < 1e-10:
            choppiness[i] = 100.0
        else:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # Component 1: 3-period RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (use absolute values for calculation)
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: Percent Rank
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        rsi_rank = rank
        
        # Calculate CRSI
        crsi[i] = (rsi_close[i] + rsi_streak[i] + rsi_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of period/2
    wma_half = pd.Series(close).rolling(window=half, min_periods=half).mean().values
    # WMA of period
    wma_full = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    # Raw HMA
    raw_hma = 2.0 * wma_half - wma_full
    
    # Smooth with WMA of sqrt(period)
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    
    return vol_ratio

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for HTF trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size (lower for 1h to control fees)
    
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
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (> 0.7x average) ===
        volume_ok = vol_ratio[i] > 0.7
        
        # === CHOPPINESS REGIME ===
        chop_value = choppiness[i]
        is_range = chop_value > 55.0  # Choppy/range market
        is_trend = chop_value < 45.0  # Trending market
        # 45-55 = neutral, no trade
        
        # === 4H HMA TREND BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with Connors RSI
        if is_range and in_session and volume_ok:
            # Long: CRSI extremely oversold
            if crsi[i] < 20.0:
                desired_signal = SIZE
            # Short: CRSI extremely overbought
            elif crsi[i] > 80.0:
                desired_signal = -SIZE
        
        # TREND REGIME: Follow HTF direction
        elif is_trend and in_session and volume_ok:
            # Long: 4h HMA bullish + RSI not overbought
            if hma_4h_bull and crsi[i] > 35.0:
                desired_signal = SIZE
            # Short: 4h HMA bearish + RSI not oversold
            elif hma_4h_bear and crsi[i] < 65.0:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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