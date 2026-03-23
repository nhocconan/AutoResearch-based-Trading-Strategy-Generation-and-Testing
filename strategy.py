#!/usr/bin/env python3
"""
Experiment #858: 30m Primary + 4h/1d HTF — Connors RSI + Dual HTF Trend + Choppiness

Hypothesis: After 562+ failed strategies, the key insight for LOWER timeframes (30m) is:
1. Use DUAL HTF trend alignment (4h HMA + 1d HMA) for signal DIRECTION
2. Use 30m Connors RSI for precise ENTRY TIMING within HTF trend
3. Add Choppiness Index to avoid entering during choppy/whipsaw periods
4. Session filter (8-20 UTC) + volume filter to reduce false signals
5. Target ONLY 30-80 trades/year (strict confluence = 3+ filters must agree)

Why this should work on 30m:
- 4h HMA(21) gives medium-term trend bias (avoids 30m noise)
- 1d HMA(21) gives secular trend filter (both must align = high probability)
- Connors RSI(3,2,100) catches short-term pullbacks within HTF trend
- Choppiness < 50 ensures we're not entering during range-bound whipsaws
- Session filter avoids low-liquidity Asian session false breakouts

Key differences from failed 30m strategies:
- DUAL HTF alignment (4h AND 1d must agree) vs single HTF
- Connors RSI instead of simple RSI (faster reaction to pullbacks)
- Choppiness filter prevents entries during choppy periods
- Conservative sizing (0.25 max) for lower TF fee sensitivity
- Strict confluence: need HTF trend + CRSI extreme + CHOP < 50 + volume

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_dual_htf_hma_chop_vol_session_atr_v1"
timeframe = "30m"
leverage = 1.0

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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    
    CRSI < 10 = extreme oversold (long opportunity)
    CRSI > 90 = extreme overbought (short opportunity)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of streaks
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    if len(streak_gain) >= streak_period:
        avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        
        avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
        avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
            streak_rsi[streak_period:] = 100 - (100 / (1 + streak_rs[streak_period:]))
    
    # Percent Rank of returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period:i]
        current_return = returns[i-1] if i > 0 else 0
        rank = np.sum(window_returns < current_return)
        percent_rank[i] = rank / len(window_returns) * 100
    
    # Combine into Connors RSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 50 as threshold for 30m timeframe.
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
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
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

def get_utc_hour(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_sma(volume, 20)
    sma_200_30m = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for secular trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract UTC hours for session filter
    utc_hours = get_utc_hour(open_time)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(chop_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_20[i]) or np.isnan(sma_200_30m[i]):
            continue
        
        # === DUAL HTF TREND ALIGNMENT (4h + 1d must agree) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Both HTF must align for high-probability signal
        dual_htf_bullish = trend_4h_bullish and trend_1d_bullish
        dual_htf_bearish = trend_4h_bearish and trend_1d_bearish
        
        # === CHOPPINESS REGIME FILTER ===
        trending_regime = chop_30m[i] < 50  # Avoid choppy markets
        ranging_regime = chop_30m[i] >= 50
        
        # === CONNORS RSI EXTREMES ===
        crsi_extreme_oversold = crsi_30m[i] < 15
        crsi_extreme_overbought = crsi_30m[i] > 85
        crsi_oversold = crsi_30m[i] < 25
        crsi_overbought = crsi_30m[i] > 75
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === SESSION FILTER (8-20 UTC = high liquidity) ===
        session_ok = 8 <= utc_hours[i] <= 20
        
        # === SMA200 SECULAR FILTER ===
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY: Dual HTF bullish + CRSI oversold + trending + volume + session ===
        if dual_htf_bullish and trending_regime:
            # Primary entry: extreme CRSI + all filters
            if crsi_extreme_oversold and volume_ok and session_ok and above_sma200:
                desired_signal = BASE_SIZE
            # Secondary entry: moderate CRSI + strong trend alignment
            elif crsi_oversold and volume_ok and trend_4h_bullish and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY: Dual HTF bearish + CRSI overbought + trending + volume + session ===
        if dual_htf_bearish and trending_regime:
            # Primary entry: extreme CRSI + all filters
            if crsi_extreme_overbought and volume_ok and session_ok and below_sma200:
                desired_signal = -BASE_SIZE
            # Secondary entry: moderate CRSI + strong trend alignment
            elif crsi_overbought and volume_ok and trend_4h_bearish and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME: Mean reversion with tighter filters ===
        if ranging_regime:
            # Only take extreme CRSI reversals in range
            if crsi_extreme_oversold and volume_ok and session_ok and above_sma200:
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought and volume_ok and session_ok and below_sma200:
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
        
        # === HOLD LOGIC — Maintain position if HTF trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if dual HTF still bullish and CRSI not overbought
                if dual_htf_bullish and crsi_30m[i] < 75:
                    desired_signal = BASE_SIZE if position_side > 0 else 0
            elif position_side < 0:
                # Hold short if dual HTF still bearish and CRSI not oversold
                if dual_htf_bearish and crsi_30m[i] > 25:
                    desired_signal = -BASE_SIZE if position_side < 0 else 0
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if dual HTF reverses bearish
            if dual_htf_bearish:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought
            if crsi_30m[i] > 90:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if dual HTF reverses bullish
            if dual_htf_bullish:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold
            if crsi_30m[i] < 10:
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