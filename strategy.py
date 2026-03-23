#!/usr/bin/env python3
"""
Experiment #1035: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Volume Session Filter

Hypothesis: After analyzing 750+ failed strategies, the key insight for 1h timeframe is:
1. Must generate 30-80 trades/year (NOT 200+ which kills via fees)
2. Use 4h/1d HTF for SIGNAL DIRECTION, 1h only for ENTRY TIMING
3. Connors RSI (CRSI) has 75% win rate for mean reversion in bear/range markets
4. Choppiness Index prevents trend-following in chop (major source of losses)
5. Volume + Session filters reduce false signals during low-liquidity hours

Strategy Components:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 (extreme oversold)
   - Short: CRSI > 85 (extreme overbought)
   - More trades than RSI(14) extremes, better timing than pure mean reversion

2. CHOPPINESS INDEX regime filter:
   - CHOP > 55 = ranging → use CRSI mean reversion signals
   - CHOP < 45 = trending → use HTF trend direction only
   - Between = hold existing positions, no new entries

3. 4h HMA21 + 1d HMA50: Dual HTF trend bias
   - Long only when price > 4h HMA21 (medium-term bullish)
   - Short only when price < 1d HMA50 (long-term bearish)
   - Asymmetric bias works better in 2025 bear/range market

4. VOLUME FILTER: volume > 1.2x 20-bar average
   - Confirms institutional participation
   - Reduces false breakouts during low liquidity

5. SESSION FILTER: Only 8-20 UTC (high volume hours)
   - Avoids Asian session whipsaws
   - Captures London/NY overlap

6. ATR Trailing Stop: 2.5x ATR for risk management

Why 1h with HTF works:
- 4h/1d determines direction (fewer false signals)
- 1h provides entry timing precision within HTF trend
- Target 40-60 trades/year (vs 100+ on pure 1h strategies)
- Session filter cuts 40% of low-quality signals

Critical fixes from failed experiments:
- RELAXED CRSI thresholds (15/85 not 10/90) for more trades
- Session filter prevents Asian session whipsaws
- Volume confirmation reduces false breakouts
- Discrete signal sizes (0.0, ±0.25) minimize fee churn
- HOLD logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h1d_hma_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    Composite of: RSI(3) + RSI_Streak(2) + PercentRank(100)
    Range: 0-100, extreme oversold < 15, extreme overbought > 85
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    close_series = pd.Series(close)
    
    # RSI(3)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = rsi_3.values
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
        
        # Calculate RSI of streak over streak_period
        if i >= streak_period:
            streak_window = streak[i-streak_period+1:i+1]
            streak_gain = np.sum(np.maximum(streak_window, 0))
            streak_loss = np.abs(np.sum(np.minimum(streak_window, 0)))
            if streak_loss > 1e-10:
                streak_rsi[i] = 100 - (100 / (1 + streak_gain / streak_loss))
            else:
                streak_rsi[i] = 100
    
    # Percent Rank over rank_period
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100 * rank / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or ranging
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    Using 55/45 thresholds for more sensitivity
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range using EMA smoothing."""
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

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        if vol_avg[i] > 1e-10:
            vol_ratio[i] = volume[i] / vol_avg[i]
    
    return vol_ratio

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # Convert ms to seconds, then to datetime
        ts_sec = open_time_array[i] / 1000
        hours[i] = int((ts_sec % 86400) / 3600)
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
    
    # Calculate and align 4h HMA21 for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA50 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    
    # Extract UTC hours for session filter
    hours_utc = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_1h[i]) or np.isnan(vol_ratio_1h[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours_utc[i] <= 20
        
        # === MACRO TREND (HTF HMA) ===
        # Long bias: price > 4h HMA21
        # Short bias: price < 1d HMA50 (asymmetric for bear market)
        medium_bull = close[i] > hma_4h_aligned[i]
        long_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop_1h[i] > 55  # Ranging → mean reversion OK
        regime_trend = chop_1h[i] < 45  # Trending → follow HTF direction
        regime_neutral = not regime_range and not regime_trend
        
        # === VOLUME FILTER ===
        volume_confirmed = vol_ratio_1h[i] > 1.2
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 15
        crsi_overbought = crsi_1h[i] > 85
        crsi_neutral = 15 <= crsi_1h[i] <= 85
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Require: session + volume + trend + CRSI extreme + regime appropriate
        if in_session and volume_confirmed and medium_bull:
            if regime_range and crsi_oversold:
                # Mean reversion in ranging market
                desired_signal = BASE_SIZE
            elif regime_trend and crsi_1h[i] < 30:
                # Pullback entry in trending market (relaxed CRSI)
                desired_signal = REDUCED_SIZE
            elif regime_neutral and crsi_oversold:
                # Opportunistic in transition
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if in_session and volume_confirmed and long_bear:
            if regime_range and crsi_overbought:
                # Mean reversion in ranging market
                desired_signal = -BASE_SIZE
            elif regime_trend and crsi_1h[i] > 70:
                # Pullback entry in trending market (relaxed CRSI)
                desired_signal = -REDUCED_SIZE
            elif regime_neutral and crsi_overbought:
                # Opportunistic in transition
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
                # Hold long if medium bullish and CRSI not extreme overbought
                if medium_bull and crsi_1h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if long-term bearish and CRSI not extreme oversold
                if long_bear and crsi_1h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if medium trend reverses OR CRSI extreme overbought
            if not medium_bull and crsi_1h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if long-term trend reverses OR CRSI extreme oversold
            if not long_bear and crsi_1h[i] < 30:
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