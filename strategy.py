#!/usr/bin/env python3
"""
Experiment #1220: 1h Primary + 4h/12h HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: 1h timeframe with strict HTF filters can achieve 30-80 trades/year while maintaining
positive Sharpe. Key insight from failures: lower TF strategies fail due to either 0 trades (too strict)
or >200 trades/year (fee drag kills profit).

Design:
- 4h HMA(21) for primary trend direction (ONLY trade WITH 4h trend)
- 12h HMA(21) for macro bias filter (avoid counter-macro trades)
- Connors RSI(3,2,100) for entry timing (looser thresholds: <25 long, >75 short)
- Choppiness Index(14) regime filter: >55 range (mean revert), <45 trend (pullback entries)
- Volume filter: volume > 0.8x 20-period average (confirm participation)
- Session filter: only 8-20 UTC (high liquidity hours)
- ATR(14) 2.5x trailing stop (tighter than 3x to cut losses faster in lower TF)
- Position size: 0.25 discrete (conservative for 1h frequency)

Why this might work:
- HTF trend filter prevents counter-trend trades (major cause of 2022 losses)
- CRSI with looser thresholds ensures >=30 trades while maintaining edge
- Session + volume filters reduce false signals during low-liquidity periods
- Choppiness regime adapts between mean-revert (range) and pullback (trend) entries

Target: Sharpe > 0.612, trades >= 30 on train, >= 3 on test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h12h_hma_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Long: CRSI < 25 (oversold), Short: CRSI > 75 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    mask = loss_3 > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_3[mask] / loss_3[mask]
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close[:rsi_period] = np.nan
    
    # RSI(streak, 2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    mask2 = streak_loss_2 > 1e-10
    rs2 = np.zeros(n)
    rs2[mask2] = streak_gain_2[mask2] / streak_loss_2[mask2]
    rsi_streak = 100.0 - (100.0 / (1.0 + rs2))
    rsi_streak[:streak_period] = np.nan
    
    # PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        if len(window) > 0:
            count_below = np.sum(window < close[i])
            percent_rank[i] = 100.0 * count_below / len(window)
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 55 = choppy/range (mean revert)
    CHOP < 45 = trending (trend follow pullbacks)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def get_hour_from_open_time(open_time_col):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = ((open_time_col // 1000) // 3600) % 24
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Volume SMA(20)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (8-20 UTC)
    hours = get_hour_from_open_time(open_time)
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_sma[i]) or atr[i] <= 1e-10 or vol_sma[i] <= 1e-10:
            continue
        
        # === HTF TREND FILTERS (4h + 12h) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        trend_12h_bull = close[i] > hma_12h_aligned[i]
        trend_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === CONNORS RSI SIGNALS (looser thresholds for trade frequency) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === CHOPPY REGIME: Mean Reversion ===
        if is_choppy:
            # Long: CRSI oversold + 4h not bearish + volume + session
            if crsi_oversold and not trend_4h_bear and volume_ok and session_ok:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + 4h not bullish + volume + session
            elif crsi_overbought and not trend_4h_bull and volume_ok and session_ok:
                desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME: Pullback Entries ===
        elif is_trending:
            # Long: 4h bullish + 12h bullish + RSI pullback (40-50) + volume + session
            rsi_pullback_long = 40.0 <= rsi[i] <= 55.0
            if trend_4h_bull and trend_12h_bull and rsi_pullback_long and volume_ok and session_ok:
                desired_signal = BASE_SIZE
            # Short: 4h bearish + 12h bearish + RSI pullback (45-60) + volume + session
            rsi_pullback_short = 45.0 <= rsi[i] <= 60.0
            if trend_4h_bear and trend_12h_bear and rsi_pullback_short and volume_ok and session_ok:
                desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE (45 <= CHOP <= 55): Use CRSI extremes only ===
        else:
            # Long: CRSI very oversold + 4h not strongly bearish
            if crsi[i] < 20.0 and not trend_4h_bear and volume_ok and session_ok:
                desired_signal = BASE_SIZE
            # Short: CRSI very overbought + 4h not strongly bullish
            elif crsi[i] > 80.0 and not trend_4h_bull and volume_ok and session_ok:
                desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals