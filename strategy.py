#!/usr/bin/env python3
"""
Experiment #1498: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: After 1100+ failed strategies, the pattern for lower TF (30m) is clear:
1. Must use VERY strict entry filters to limit trades to 30-80/year (fee drag killer)
2. HTF (4h/1d) for DIRECTION, 30m only for ENTRY TIMING
3. Choppiness Index regime filter: CHOP>55=range(mean revert), CHOP<45=trend(follow)
4. Connors RSI (CRSI) for precise mean reversion entries (75% win rate in literature)
5. Session filter (8-20 UTC) avoids low-liquidity whipsaws
6. Volume filter confirms institutional participation

Key design choices:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- CHOP(14) > 55 = range regime → mean revert at CRSI extremes
- CHOP(14) < 45 = trend regime → follow 4h HMA direction
- 4h HMA(21) + 1d HMA(21) for macro trend bias (dual HTF confirmation)
- Session: only trade 8-20 UTC (major market hours)
- Volume: > 0.8x 20-bar average
- ATR(14) 2.5x trailing stop
- Position size: 0.25 (smaller for 30m to reduce fee impact)

Timeframe: 30m (as required)
HTF: 4h + 1d (call get_htf_data ONCE each before loop!)
Position Size: 0.25 (discrete: 0.0, ±0.25)
Target: 40-80 trades/train, 5-15 trades/test, Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_4h1d_hma_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * (ATR(14) sum / (Highest High - Lowest Low)) * 100 / log10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr_vals = calculate_atr(high, low, close, period)
    
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 1e-10:
            continue
        
        atr_sum = np.nansum(atr_vals[i - period + 1:i + 1])
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10:
            chop[i] = 100.0 * (atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive up/down days
    PercentRank: percentile of today's change vs last 100 days
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3) of price
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if delta[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if delta[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive for RSI calculation
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0.0)
    streak_loss = np.where(streak < 0, np.abs(streak), 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_smooth > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask] / streak_loss_smooth[mask]))
    rsi_streak[streak_loss_smooth <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # Component 3: PercentRank of daily returns
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        current_return = returns[i]
        rank = np.sum(window < current_return)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine components
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + rsi_streak[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_sma(close, period=20):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def extract_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) // 3600) % 24
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
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    sma_vol = calculate_sma(volume, period=20)
    hour = extract_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 30m to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need more warmup for CRSI
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_vol[i]) or sma_vol[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= hour[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * sma_vol[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = choppiness[i] > 55.0  # Range/mean-revert regime
        is_trend = choppiness[i] < 45.0  # Trending regime
        
        # === MACRO TREND (4h + 1d HMA) ===
        # Both HTF must agree for strong signal
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bull: both 4h and 1d HMA bullish
        strong_bull = hma_4h_bull and hma_1d_bull
        # Strong bear: both 4h and 1d HMA bearish
        strong_bear = hma_4h_bear and hma_1d_bear
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        crsi_moderate_oversold = crsi[i] < 25.0
        crsi_moderate_overbought = crsi[i] > 75.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at CRSI extremes
        if in_session and volume_ok and is_range:
            # Long: CRSI very oversold + HTF not strongly bearish
            if crsi_oversold and not strong_bear:
                desired_signal = BASE_SIZE
            # Short: CRSI very overbought + HTF not strongly bullish
            elif crsi_overbought and not strong_bull:
                desired_signal = -BASE_SIZE
            # Moderate entries with HTF confirmation
            elif crsi_moderate_oversold and strong_bull:
                desired_signal = BASE_SIZE * 0.8
            elif crsi_moderate_overbought and strong_bear:
                desired_signal = -BASE_SIZE * 0.8
        
        # TREND REGIME: Follow HTF direction on pullbacks
        elif in_session and volume_ok and is_trend:
            # Long pullback in uptrend
            if strong_bull and crsi_moderate_oversold:
                desired_signal = BASE_SIZE
            # Short pullback in downtrend
            elif strong_bear and crsi_moderate_overbought:
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
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.8
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