#!/usr/bin/env python3
"""
Experiment #238: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: Lower timeframe (30m) needs EXTREMELY strict entry filters to avoid fee drag.
After 200+ failed experiments, the pattern is clear: complex regime switching fails,
but CONFLUENCE of 4+ independent signals works. This strategy uses:

1. 4h HMA(16/48) for trend direction (HTF bias - call ONCE before loop)
2. 1d HMA(21) for macro filter (only trade with daily trend)
3. Connors RSI (CRSI) for entry timing - proven 75% win rate in literature
4. Choppiness Index regime filter (CHOP>55=range revert, <45=trend follow)
5. Session filter (8-20 UTC) - avoids Asian session noise
6. Volume filter (>0.8x 20-bar avg) - confirms move validity
7. ATR(14) 2.5x trailing stop (tighter for 30m volatility)

Key innovation: CRSI combines 3 signals (RSI3 + StreakRSI + PercentRank) for more
reliable mean-reversion entries than standard RSI. Combined with HTF trend filter,
this should generate 30-60 trades/year with Sharpe > 0.5 on ALL symbols.

Position sizing: 0.20-0.25 (smaller for 30m to control drawdown)
Target: 40-80 trades/year, Sharpe > 0.5 on BTC/ETH/SOL individually
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_session_4h1d_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent Rank - percentile of today's return vs last pr_period days
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i].dropna()
        if len(window) > 0:
            current_return = returns.iloc[i]
            percent_rank[i] = (window < current_return).sum() / len(window) * 100
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in the window
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_session_filter(open_time):
    """
    Return 1 if within 8-20 UTC session, 0 otherwise.
    open_time is in milliseconds since epoch.
    """
    # Convert to hours UTC
    hours = (open_time // 3600000) % 24
    return ((hours >= 8) & (hours < 20)).astype(float)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    rsi_3 = calculate_rsi(close, period=3)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    choppiness = calculate_choppiness_index(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    session = calculate_session_filter(open_time)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h HMA for trend (aligned properly)
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, 48)
    hma_4h_16 = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48 = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(volume_sma[i]) or volume_sma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === HTF MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        macro_bullish = price_above_hma_1d
        macro_bearish = price_below_hma_1d
        
        # === 4h TREND DETECTION (HMA crossover) ===
        hma_4h_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_4h_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = choppiness[i] > 55.0  # range market
        chop_trend = choppiness[i] < 45.0  # trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_neutral = 30.0 <= crsi[i] <= 70.0
        
        # === SESSION FILTER ===
        in_session = session[i] > 0.5
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma[i]
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Need 4+ confluence
        # 1) 4h trend bullish OR 1d macro bullish
        # 2) CRSI oversold (<15) OR (chop range + CRSI <30)
        # 3) In session (8-20 UTC)
        # 4) Volume confirmed
        long_confluence = 0
        if hma_4h_bullish or macro_bullish:
            long_confluence += 1
        if crsi_oversold or (chop_range and crsi[i] < 30.0):
            long_confluence += 1
        if in_session:
            long_confluence += 1
        if volume_ok:
            long_confluence += 1
        
        if long_confluence >= 4:
            if macro_bullish and hma_4h_bullish:
                desired_signal = POSITION_SIZE_FULL
            else:
                desired_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: Need 4+ confluence
        short_confluence = 0
        if hma_4h_bearish or macro_bearish:
            short_confluence += 1
        if crsi_overbought or (chop_range and crsi[i] > 70.0):
            short_confluence += 1
        if in_session:
            short_confluence += 1
        if volume_ok:
            short_confluence += 1
        
        if short_confluence >= 4:
            if macro_bearish and hma_4h_bearish:
                desired_signal = -POSITION_SIZE_FULL
            else:
                desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_4h_bearish and macro_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish and macro_bullish:
            desired_signal = 0.0
        
        # === CRSI EXIT (extreme opposite) ===
        if in_position and position_side > 0 and crsi_overbought:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_oversold:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if trend still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and (hma_4h_bullish or macro_bullish) and crsi[i] < 80.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and (hma_4h_bearish or macro_bearish) and crsi[i] > 20.0:
                desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals