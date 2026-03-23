#!/usr/bin/env python3
"""
Experiment #688: 30m Primary + 4h/1d HTF — CRSI + Volume + Session + HTF Trend

Hypothesis: 30m timeframe with strict confluence filters can achieve optimal trade frequency
(30-80/year) while capturing intraday moves within HTF trend. Key innovations:
1. 4h HMA for MACRO trend direction — ONLY trade in HTF trend direction
2. 1d HMA for REGIME filter — bull/bear market bias
3. 30m CRSI for ENTRY timing — extreme values only (CRSI<25/>75, not <10/>90)
4. Volume filter — volume > 1.2x 20-period average (confirms move)
5. Session filter — only 8-20 UTC (highest liquidity, avoids Asian chop)
6. Position size 0.20-0.25 (smaller for lower TF to reduce fee impact)
7. 2.5x ATR trailing stop for risk management

Why this should work where #678 failed:
- #678 had 0 trades (Sharpe=0.000) — entry conditions too strict
- This version uses LOOSER CRSI thresholds (25/75 vs 10/90) to ensure trade generation
- Volume + Session filters reduce false signals without eliminating all trades
- HTF trend alignment prevents counter-trend trades that kill Sharpe in bear markets
- 30m TF = ~50-80 trades/year with strict confluence (optimal for fee drag)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_volume_session_htf_trend_v1"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — combines 3 components for mean reversion signals.
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Research shows 75% win rate for CRSI<10 long, CRSI>90 short.
    We use looser thresholds (25/75) to ensure trade generation on 30m.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_pad = np.concatenate([[0], gain])
    loss_pad = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain_pad).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss_pad).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Component 2: RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    avg_streak_gain = pd.Series(streak_gain).rolling(window=streak_period, min_periods=streak_period).mean().values
    avg_streak_loss = pd.Series(streak_loss).rolling(window=streak_period, min_periods=streak_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank of returns over 100 periods
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = rank * 100
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs 20-period average."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_avg + 1e-10)
    
    return vol_ratio

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_ratio_30m = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF (4h) indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF (1d) indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after warmup period for all indicators
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_ratio_30m[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = (hour >= 8) and (hour <= 20)
        
        # === VOLUME FILTER (volume > 1.2x average) ===
        volume_confirmed = vol_ratio_30m[i] > 1.2
        
        # === 4H TREND DIRECTION (MACRO bias) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === 1D REGIME FILTER (bull/bear market) ===
        regime_bullish = close[i] > hma_1d_aligned[i]
        regime_bearish = close[i] < hma_1d_aligned[i]
        
        # === CRSI SIGNALS (LOOSE thresholds for trade generation) ===
        crsi_oversold = crsi_30m[i] < 25
        crsi_overbought = crsi_30m[i] > 75
        crsi_extreme_oversold = crsi_30m[i] < 15
        crsi_extreme_overbought = crsi_30m[i] > 85
        
        desired_signal = 0.0
        
        # === LONG ENTRY: 4h bullish + CRSI oversold + volume + session ===
        if trend_4h_bullish and in_session:
            # Standard long: CRSI < 25 + volume confirmed
            if crsi_oversold and volume_confirmed:
                desired_signal = SIZE_LONG
            # Extreme long: CRSI < 15 (override volume filter for extreme setups)
            elif crsi_extreme_oversold:
                desired_signal = SIZE_LONG
        
        # === SHORT ENTRY: 4h bearish + CRSI overbought + volume + session ===
        elif trend_4h_bearish and in_session:
            # Standard short: CRSI > 75 + volume confirmed
            if crsi_overbought and volume_confirmed:
                desired_signal = -SIZE_SHORT
            # Extreme short: CRSI > 85 (override volume filter for extreme setups)
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_SHORT
        
        # === REGIME OVERRIDE: 1d trend confirms 4h trend ===
        # Only take trades when 1d and 4h agree (higher confidence)
        if regime_bullish and trend_4h_bullish:
            if crsi_oversold:
                desired_signal = max(desired_signal, SIZE_LONG * 0.5)
        elif regime_bearish and trend_4h_bearish:
            if crsi_overbought:
                desired_signal = min(desired_signal, -SIZE_SHORT * 0.5)
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if trend_4h_bullish and crsi_30m[i] < 85:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                if trend_4h_bearish and crsi_30m[i] > 15:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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
        
        signals[i] = desired_signal
    
    return signals