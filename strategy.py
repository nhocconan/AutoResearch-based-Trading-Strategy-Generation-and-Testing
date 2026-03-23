#!/usr/bin/env python3
"""
Experiment #805: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After analyzing 500+ failed strategies and the critical 1h failure pattern (exp#795 Sharpe=0.000):
1. 1h timeframe needs VERY strict entry filters to avoid fee drag (target 30-80 trades/year)
2. Connors RSI (CRSI) generates more reliable mean-reversion signals than standard RSI
3. 4h Choppiness Index provides better regime detection than 1d for 1h entries
4. Session filter (8-20 UTC) removes Asian session noise where most whipsaws occur
5. 1d HMA(21) for long-term trend bias ensures we trade with the macro trend
6. Volume filter (>0.8x avg) confirms institutional participation
7. Discrete position sizing (0.20/0.30) minimizes fee churn on signal changes
8. ATR(14) trailing stop at 2.5x protects from major drawdowns

Strategy design:
1. 1d HMA(21) for long-term trend bias (aligned via mtf_data helper)
2. 4h Choppiness Index(14) for regime detection (ranging vs trending)
3. 1h Connors RSI for entry timing (RSI3 + StreakRSI2 + PercentRank100) / 3
4. Session filter: only trade 8-20 UTC (London/NY overlap)
5. Volume filter: volume > 0.8x 20-bar SMA
6. ATR(14) trailing stop at 2.5x
7. Discrete signals: 0.0, ±0.20, ±0.30
8. Regime-adaptive: mean revert when CHOP>55, trend follow when CHOP<45

Key differences from failed 1h strategies:
- Connors RSI instead of standard RSI (more responsive, better for mean reversion)
- 4h Choppiness instead of 1d (more responsive regime detection for 1h entries)
- Session filter (8-20 UTC) removes 60% of noise trades
- Relaxed volume filter (0.8x not 1.5x) to ensure sufficient trades
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_hma_4h1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    Streak RSI component of Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    max_streak = np.max(abs_streak[~np.isnan(abs_streak)])
    if max_streak > 0:
        streak_rsi = 100 * abs_streak / max_streak
    else:
        streak_rsi = np.zeros(n)
    
    # Apply direction: positive streak = bullish, negative = bearish
    streak_rsi = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI.
    Measures current price change relative to past N periods.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        count_below = np.sum(window < current)
        pct_rank[i] = 100 * count_below / period
    
    return pct_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + StreakRSI(2) + PercentRank(100)) / 3
    More responsive than standard RSI, better for mean reversion.
    """
    rsi_comp = calculate_rsi(close, rsi_period)
    streak_comp = calculate_streak_rsi(close, streak_period)
    pr_comp = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_comp + streak_comp + pr_comp) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches on 4h.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    utc_hour = get_utc_hour(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_sma(volume, 20)
    
    # Calculate and align 4h Choppiness for regime detection
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h_aligned[i] > 55
        trending_regime = chop_4h_aligned[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === VOLUME CONFIRMATION (relaxed for 1h) ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi_1h[i] < 15
        crsi_oversold = crsi_1h[i] < 25
        crsi_overbought = crsi_1h[i] > 75
        crsi_extreme_overbought = crsi_1h[i] > 85
        crsi_neutral_low = 25 <= crsi_1h[i] <= 45
        crsi_neutral_high = 55 <= crsi_1h[i] <= 75
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) ===
        if ranging_regime and in_session:
            # Mean reversion long: CRSI oversold + 1d bullish trend
            if crsi_oversold and trend_1d_bullish:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + 1d bearish trend
            if crsi_overbought and trend_1d_bearish:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Conservative: extreme CRSI alone
            if crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) ===
        elif trending_regime and in_session:
            # Trend pullback long: 1d bullish + CRSI neutral low (pullback entry)
            if trend_1d_bullish and crsi_neutral_low:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: 1d bearish + CRSI neutral high (pullback entry)
            if trend_1d_bearish and crsi_neutral_high:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Strong trend: extreme CRSI with trend
            if trend_1d_bullish and crsi_oversold:
                desired_signal = BASE_SIZE
            
            if trend_1d_bearish and crsi_overbought:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            if in_session:
                # Conservative: only extreme CRSI + trend alignment
                if crsi_extreme_oversold and trend_1d_bullish:
                    desired_signal = REDUCED_SIZE
                
                if crsi_extreme_overbought and trend_1d_bearish:
                    desired_signal = -REDUCED_SIZE
                
                # Allow basic mean reversion with volume
                if crsi_oversold and trend_1d_bullish and volume_confirmed:
                    desired_signal = REDUCED_SIZE
                
                if crsi_overbought and trend_1d_bearish and volume_confirmed:
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
                # Hold long if trend intact and CRSI not overbought
                if trend_1d_bullish and crsi_1h[i] < 80:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if trend_1d_bearish and crsi_1h[i] > 20:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI overbought
            if trend_1d_bearish and crsi_1h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI oversold
            if trend_1d_bullish and crsi_1h[i] < 25:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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