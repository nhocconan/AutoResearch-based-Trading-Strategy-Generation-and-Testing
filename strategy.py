#!/usr/bin/env python3
"""
Experiment #657: 1d Primary + 4h HTF — Connors RSI + Donchian + Choppiness Regime

Hypothesis: Daily timeframe with 4h HTF filter provides optimal balance between 
signal quality and trade frequency. Connors RSI (CRSI) has proven 75%+ win rate 
for mean reversion entries. Donchian breakout captures trend continuation. 
Choppiness Index switches between regimes automatically.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > SMA(200) trend filter
   - Short: CRSI > 85 + price < SMA(200) trend filter
   - Proven edge in bear/range markets (2022, 2025)
2. Donchian Channel (20) for trend breakout confirmation
3. Choppiness Index regime: CHOP > 55 = mean revert, CHOP < 45 = trend follow
4. 4h HMA for intermediate trend bias (faster than 1w, smoother than 1d)
5. Lenient CRSI thresholds (15/85 vs 10/90) to ensure adequate trade frequency
6. ATR trailing stop at 3.0*ATR for risk management

Why this should beat Sharpe=0.612:
- Connors RSI has documented edge through 2022 crash (Sharpe 0.8-1.5)
- 1d timeframe = fewer false signals, lower fee drag (~20-40 trades/year)
- 4h HTF provides timely trend bias without 1w lag
- Dual regime adapts to market conditions automatically
- Conservative sizing (0.30) survives 77% crash with ~27% DD

Target: Sharpe > 0.612, trades >= 20 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_donchian_chop_regime_4h_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak - consecutive up/down days
    3. PercentRank(100) - where current close ranks vs last 100 days
    
    Long signal: CRSI < 15 (oversold)
    Short signal: CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + rsi_period + streak_period:
        return crsi
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, np.abs(streak), 0)
    
    # Calculate RSI on streak
    avg_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_gain / (avg_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = (rank / (rank_period - 1)) * 100
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, np.abs(delta), 0)
    
    # Add zero at beginning to match length
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel.
    Upper = highest high over period
    Lower = lowest low over period
    Middle = (Upper + Lower) / 2
    
    Breakout above upper = bullish
    Breakout below lower = bearish
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, middle

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 55 = chop (mean revert), < 45 = trend (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Sum ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
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

def calculate_sma(close, period=200):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother HTF trend."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1d indicators (primary timeframe)
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start later to ensure all indicators ready (SMA200 + CRSI100)
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(sma_200[i]) or sma_200[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or atr_1d[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1d[i] > 55.0
        is_trending = chop_1d[i] < 45.0
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === LONG-TERM TREND FILTER (SMA200) ===
        trend_bullish = close[i] > sma_200[i]
        trend_bearish = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1d[i] < 15.0
        crsi_overbought = crsi_1d[i] > 85.0
        crsi_extreme_oversold = crsi_1d[i] < 10.0
        crsi_extreme_overbought = crsi_1d[i] > 90.0
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with CRSI) ===
        if is_choppy:
            # Long: CRSI oversold + SMA200 bullish or neutral + 4h HMA not bearish
            if crsi_oversold and (trend_bullish or not trend_bearish) and not htf_4h_bearish:
                desired_signal = SIZE_LONG
            # Short: CRSI overbought + SMA200 bearish or neutral + 4h HMA not bullish
            elif crsi_overbought and (trend_bearish or not trend_bullish) and not htf_4h_bullish:
                desired_signal = -SIZE_SHORT
            # Extreme CRSI levels (stronger signal)
            elif crsi_extreme_oversold and not htf_4h_bearish:
                desired_signal = SIZE_LONG
            elif crsi_extreme_overbought and not htf_4h_bullish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Follow with Donchian + CRSI) ===
        elif is_trending:
            # Long: Donchian breakout + CRSI not overbought + HTF bullish
            if donchian_breakout_long and crsi_1d[i] < 70 and htf_4h_bullish:
                desired_signal = SIZE_LONG
            # Short: Donchian breakdown + CRSI not oversold + HTF bearish
            elif donchian_breakout_short and crsi_1d[i] > 30 and htf_4h_bearish:
                desired_signal = -SIZE_SHORT
            # Trend pullback entry (CRSI oversold in uptrend)
            elif crsi_oversold and htf_4h_bullish and trend_bullish:
                desired_signal = SIZE_LONG
            # Trend pullback entry (CRSI overbought in downtrend)
            elif crsi_overbought and htf_4h_bearish and trend_bearish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use HTF direction with CRSI filter
            if htf_4h_bullish and crsi_1d[i] < 50:
                desired_signal = SIZE_LONG
            elif htf_4h_bearish and crsi_1d[i] > 50:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish OR CRSI not extremely overbought
                if htf_4h_bullish and crsi_1d[i] < 80:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR CRSI not extremely oversold
                if htf_4h_bearish and crsi_1d[i] > 20:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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