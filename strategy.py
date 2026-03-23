#!/usr/bin/env python3
"""
Experiment #975: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After 700+ failed strategies, the key is balancing trade frequency with quality.
1h timeframe needs VERY strict filters to avoid fee drag (>100 trades/year = failure).

Key innovations:
1. Connors RSI (CRSI) for precise mean-reversion entries: CRSI<15 long, CRSI>85 short
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   This catches oversold/overbought extremes better than standard RSI(14)
2. Choppiness Index regime filter: CHOP>55 = range (mean revert), CHOP<45 = trend (follow)
3. 4h HMA(21) for medium-term trend bias — only trade WITH HTF trend
4. 1d HMA(21) for macro regime — avoid counter-macro trades
5. Session filter: ONLY 8-20 UTC (high liquidity, avoid Asian session whipsaw)
6. Volume filter: volume > 0.8x 20-period average (confirm interest)
7. Discrete signals: 0.0, ±0.20, ±0.30 to minimize fee churn
8. ATR(14) stoploss at 2.5x — mandatory risk management

Why this should work:
- CRSI has 75% win rate in research (Larry Connors)
- Session filter reduces noise from low-liquidity hours
- HTF trend filter prevents counter-trend disasters (2022 crash)
- Target: 40-70 trades/year on 1h (fee drag ~2-3.5%)

Timeframe: 1h (target 40-70 trades/year)
Position size: 0.20-0.30 (conservative for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_4h1d_hma_volume_atr_v1"
timeframe = "1h"
leverage = 1.0

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

def calculate_crsi(close):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days (period=2)
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < 100:
        return crsi
    
    # RSI(3)
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (period=2)
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
    avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank (100)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    percent_rank = np.full(n, np.nan)
    for i in range(100, n):
        window = returns[i-99:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100
    
    # Combine CRSI
    for i in range(100, n):
        if not np.isnan(rsi3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    vol_avg_1h = calculate_volume_avg(volume, period=20)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
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
        if np.isnan(chop_1h[i]) or np.isnan(vol_avg_1h[i]) or vol_avg_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # Extract UTC hour for session filter
        hour = get_hour_from_open_time(open_time[i])
        
        # === SESSION FILTER: Only trade 8-20 UTC (high liquidity) ===
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER: volume > 0.8x average ===
        volume_ok = volume[i] > 0.8 * vol_avg_1h[i]
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 15
        crsi_overbought = crsi_1h[i] > 85
        crsi_extreme_oversold = crsi_1h[i] < 10
        crsi_extreme_overbought = crsi_1h[i] > 90
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime and in_session and volume_ok:
            # Long: CRSI oversold + HTF trend support
            if crsi_oversold and (macro_bull or trend_4h_bullish):
                desired_signal = BASE_SIZE
            # Long: CRSI extreme oversold (override trend filter)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + HTF trend support
            if crsi_overbought and (macro_bear or trend_4h_bearish):
                desired_signal = -BASE_SIZE
            # Short: CRSI extreme overbought (override trend filter)
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with Pullback ===
        elif trending_regime and in_session and volume_ok:
            # Long: Bullish trend + CRSI pullback
            if macro_bull or trend_4h_bullish:
                if crsi_oversold:
                    desired_signal = BASE_SIZE
                elif crsi_1h[i] < 30:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + CRSI rally
            if macro_bear or trend_4h_bearish:
                if crsi_overbought:
                    desired_signal = -BASE_SIZE
                elif crsi_1h[i] > 70:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only extreme CRSI with HTF confluence
            if crsi_extreme_oversold and (macro_bull or trend_4h_bullish) and in_session:
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought and (macro_bear or trend_4h_bearish) and in_session:
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
                if (macro_bull or trend_4h_bullish) and crsi_1h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (macro_bear or trend_4h_bearish) and crsi_1h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought
            if crsi_1h[i] > 80:
                desired_signal = 0.0
            # Exit if macro + medium trend reverses
            if macro_bear and trend_4h_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold
            if crsi_1h[i] < 20:
                desired_signal = 0.0
            # Exit if macro + medium trend reverses
            if macro_bull and trend_4h_bullish:
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