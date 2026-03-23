#!/usr/bin/env python3
"""
Experiment #983: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + Donchian

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance of
signal quality vs trade frequency. Connors RSI (proven 75% win rate) for entries,
Choppiness Index for regime detection, Donchian for trend confirmation.

Key insights from research:
1. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI<15 + price>SMA200. Short: CRSI>85 + price<SMA200
2. Choppiness Index: CHOP>55 = range (mean revert), CHOP<45 = trend (breakout)
3. 1w HMA(21) for macro trend bias — only trade in direction of weekly trend
4. Donchian(20) breakout for trend entries, BB-style for mean reversion
5. ATR(14) 2.5x trailing stoploss for risk management

Why 1d timeframe:
- Target 20-50 trades/year (optimal fee drag: 1-2.5%)
- Weekly HTF provides strong macro filter
- Less noise than lower TF, clearer regime signals
- Proven to work in both bull (2021) and bear (2022, 2025) markets

Critical improvements over failed experiments:
- RELAXED CRSI thresholds (15/85 not 10/90) to ensure minimum trades
- Dual regime logic adapts to market conditions
- Weekly HMA filter prevents counter-trend trades
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_donchian_1w_hma_regime_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors Relative Strength Index.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_fast = 100 - (100 / (1 + rs))
    rsi_fast = np.clip(rsi_fast, 0, 100)
    
    # Component 2: RSI of Up/Down Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_series = pd.Series(streak_gain)
    streak_loss_series = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = streak_loss_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: Percent Rank of close over lookback
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1)
        percent_rank[i] = rank * 100
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = range, CHOP < 38.2 = strong trend
    We use 55/45 thresholds for regime switching.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(series, period):
    """Hull Moving Average — smoother and more responsive than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        prev_close = close[i-1]
        tr[i] = max(high[i] - low[i], np.abs(high[i] - prev_close), np.abs(low[i] - prev_close))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(close := high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # === LONG-TERM TREND (SMA200) ===
        long_term_bullish = close[i] > sma_200[i]
        long_term_bearish = close[i] < sma_200[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop[i] > 55
        trending_regime = chop[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donch_upper[i]
        donchian_breakout_short = close[i] < donch_lower[i]
        near_donch_lower = (close[i] - donch_lower[i]) / (donch_upper[i] - donch_lower[i]) < 0.15 if (donch_upper[i] - donch_lower[i]) > 1e-10 else False
        near_donch_upper = (close[i] - donch_lower[i]) / (donch_upper[i] - donch_lower[i]) > 0.85 if (donch_upper[i] - donch_lower[i]) > 1e-10 else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + long-term bullish bias
            if crsi_oversold and long_term_bullish:
                desired_signal = BASE_SIZE
            # Long: CRSI extreme oversold (any regime)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            # Long: Near Donchian lower + weekly bullish
            elif near_donch_lower and weekly_bullish:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + long-term bearish bias
            if crsi_overbought and long_term_bearish:
                desired_signal = -BASE_SIZE
            # Short: CRSI extreme overbought (any regime)
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            # Short: Near Donchian upper + weekly bearish
            elif near_donch_upper and weekly_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Donchian breakout + weekly bullish confirmation
            if donchian_breakout_long and weekly_bullish:
                desired_signal = BASE_SIZE
            # Long: Pullback in uptrend + CRSI recovering
            elif weekly_bullish and long_term_bullish and crsi[i] < 40 and crsi[i] > crsi[i-1] if i > 0 else False:
                desired_signal = REDUCED_SIZE
            
            # Short: Donchian breakdown + weekly bearish confirmation
            if donchian_breakout_short and weekly_bearish:
                desired_signal = -BASE_SIZE
            # Short: Rally in downtrend + CRSI weakening
            elif weekly_bearish and long_term_bearish and crsi[i] > 60 and crsi[i] < crsi[i-1] if i > 0 else False:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only extreme CRSI with trend confirmation
            if crsi_extreme_oversold and (weekly_bullish or long_term_bullish):
                desired_signal = BASE_SIZE
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (weekly_bearish or long_term_bearish):
                desired_signal = -BASE_SIZE
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly trend intact and CRSI not overbought
                if weekly_bullish and crsi[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly trend intact and CRSI not oversold
                if weekly_bearish and crsi[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly + long-term trend reverses + CRSI overbought
            if weekly_bearish and long_term_bearish and crsi[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly + long-term trend reverses + CRSI oversold
            if weekly_bullish and long_term_bullish and crsi[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals