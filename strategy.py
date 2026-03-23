#!/usr/bin/env python3
"""
Experiment #647: 1d Primary + 4h HTF — Connors RSI + Donchian Breakout + Choppiness Regime

Hypothesis: Daily timeframe with 4h HTF filter provides optimal balance between signal 
quality and trade frequency. Connors RSI excels at mean reversion in choppy markets 
(75% win rate documented), while Donchian breakouts capture trends. Choppiness Index 
switches between regimes automatically.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI for reversal entries
   - Long: CRSI < 15, Short: CRSI > 85 (looser than extreme for trade frequency)
2. Donchian Channel(20) breakout for trend following
3. Choppiness Index regime: CHOP > 55 = mean revert, CHOP < 45 = trend follow
4. 4h HMA for intermediate trend bias (not too slow like 1w, not too fast like 1h)
5. SMA(200) filter for macro trend direction
6. Conservative sizing (0.28 long, 0.25 short) to survive crashes

Why this should beat Sharpe=0.612:
- CRSI has documented edge in bear/range markets (2022, 2025)
- 1d timeframe = fewer false signals, lower fee drag (~30-50 trades/year)
- 4h HTF provides timely trend bias without lag of 1w
- Dual regime adapts to market conditions automatically
- Looser CRSI thresholds ensure adequate trade frequency (critical lesson from 0-trade failures)

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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) - short-term momentum
    2. RSI of streak duration - measures consecutive up/down days
    3. PercentRank - where current close ranks in last 100 days
    
    Long signal: CRSI < 15 (oversold)
    Short signal: CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad to match length
    gain = np.pad(gain, (1, 0), mode='constant', constant_values=0)
    loss = np.pad(loss, (1, 0), mode='constant', constant_values=0)
    
    # EMA for RSI
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
        rsi_short = np.clip(rsi_short, 0, 100)
    
    # Component 2: RSI of streak duration
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Absolute streak for RSI calculation
    abs_streak = np.abs(streak)
    
    # RSI of streak (up streaks = gain, down streaks = loss conceptually)
    streak_gain = np.where(streak > 0, abs_streak, 0)
    streak_loss = np.where(streak < 0, abs_streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
        rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = (rank / rank_period) * 100
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel.
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = long signal
    Breakout below lower = short signal
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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

def calculate_sma(close, period=200):
    """Simple Moving Average for macro trend filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1d indicators (primary timeframe)
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.28
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start later to ensure all indicators ready (SMA200 + CRSI rank_period)
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]):
            continue
        if np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1d[i] > 55.0
        is_trending = chop_1d[i] < 45.0
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === MACRO TREND (SMA200) ===
        macro_bullish = close[i] > sma_200[i]
        macro_bearish = close[i] < sma_200[i]
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_1d[i] < 15.0
        crsi_overbought = crsi_1d[i] > 85.0
        
        # === DONCHIAN BREAKOUT (Trend Following) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with CRSI) ===
        if is_choppy:
            # Long: CRSI oversold + HTF 4h not bearish + price near/above SMA200
            if crsi_oversold and not htf_4h_bearish:
                desired_signal = SIZE_LONG
            # Short: CRSI overbought + HTF 4h not bullish + price near/below SMA200
            elif crsi_overbought and not htf_4h_bullish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Follow with Donchian) ===
        elif is_trending:
            # Long: Donchian breakout + HTF bullish + macro bullish
            if donchian_breakout_long and htf_4h_bullish and macro_bullish:
                desired_signal = SIZE_LONG
            # Short: Donchian breakout + HTF bearish + macro bearish
            elif donchian_breakout_short and htf_4h_bearish and macro_bearish:
                desired_signal = -SIZE_SHORT
            # CRSI pullback entry in trend
            elif htf_4h_bullish and macro_bullish and crsi_oversold:
                desired_signal = SIZE_LONG
            elif htf_4h_bearish and macro_bearish and crsi_overbought:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION (45 <= CHOP <= 55) ===
        else:
            # Use HTF direction with CRSI filter for entry timing
            if htf_4h_bullish and crsi_1d[i] < 40:
                desired_signal = SIZE_LONG
            elif htf_4h_bearish and crsi_1d[i] > 60:
                desired_signal = -SIZE_SHORT
        
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
                # Hold long if HTF still bullish and CRSI not extremely overbought
                if htf_4h_bullish and crsi_1d[i] < 80:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish and CRSI not extremely oversold
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