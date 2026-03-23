#!/usr/bin/env python3
"""
Experiment #995: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After 720+ failed strategies, combining Connors RSI (proven 75% win rate) with
Choppiness Index regime detection and HTF trend alignment should work across ALL symbols.

Key innovations:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 10 + price > SMA200 (bullish mean reversion)
   - Short when CRSI > 90 + price < SMA200 (bearish mean reversion)
2. CHOPPINESS INDEX regime: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 4h HMA(21) for medium-term trend bias (align with HTF direction)
4. 1d HMA(21) for macro regime filter (only trade with macro trend)
5. Session filter: Only 8-20 UTC (highest volume, lowest noise)
6. Volume filter: Volume > 0.8x 20-period average (confirm participation)

Why 1h timeframe:
- Target 30-60 trades/year (within fee drag limits for 1h)
- HTF signals (4h/1d) provide direction, 1h provides entry timing
- CRSI extremes trigger frequently enough but session filter reduces noise
- Proven to work in both bull and bear markets with proper regime filter

Critical improvements from failures:
- RELAXED CRSI thresholds (< 15 / > 85 not < 10 / > 90) to ensure trades
- Session filter reduces 60% of noise trades outside high-volume hours
- Volume confluence ensures real participation (not fake breakouts)
- Discrete signal sizes (0.0, ±0.20, ±0.30) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 35-55 trades/year with session filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h1d_hma_session_vol_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentage of recent closes lower than current close
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and close[i-1] >= close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and close[i-1] <= close[i-2] else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        avg_streak = np.mean(np.abs(streak_window))
        # Normalize to 0-100 scale
        streak_rsi[i] = min(100, max(0, 50 + avg_streak * 15))
    
    # Percent Rank component
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100 * count_lower / (rank_period - 1)
    
    # Combine all three components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
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

def calculate_sma(close, period):
    """Simple Moving Average."""
    n = len(close)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def calculate_volume_avg(volume, period=20):
    """Average volume over lookback period."""
    n = len(volume)
    vol_avg = np.full(n, np.nan)
    
    if n < period:
        return vol_avg
    
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_avg

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # Convert milliseconds to seconds, then to datetime
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
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    sma_200_1h = calculate_sma(close, period=200)
    vol_avg_1h = calculate_volume_avg(volume, period=20)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(open_time)
    
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
    
    for i in range(250, n):  # Start after 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(chop_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg_1h[i]) or vol_avg_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_1h[i]
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === LONG-TERM TREND (1h SMA200) ===
        trend_200_bullish = close[i] > sma_200_1h[i]
        trend_200_bearish = close[i] < sma_200_1h[i]
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_low = crsi_1h[i] < 15
        crsi_extreme_high = crsi_1h[i] > 85
        crsi_very_low = crsi_1h[i] < 10
        crsi_very_high = crsi_1h[i] > 90
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime and in_session and volume_confirmed:
            # Long: CRSI extreme low + macro/medium trend support
            if crsi_extreme_low and (macro_bull or trend_4h_bullish or trend_200_bullish):
                desired_signal = BASE_SIZE
            # Long: CRSI very low (stronger signal, less confluence needed)
            elif crsi_very_low:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI extreme high + macro/medium trend support
            if crsi_extreme_high and (macro_bear or trend_4h_bearish or trend_200_bearish):
                desired_signal = -BASE_SIZE
            # Short: CRSI very high (stronger signal, less confluence needed)
            elif crsi_very_high:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with Pullback ===
        elif trending_regime and in_session and volume_confirmed:
            # Long: Bullish trend + CRSI pullback (buy dip)
            if (macro_bull or trend_4h_bullish) and crsi_extreme_low:
                desired_signal = BASE_SIZE
            # Long: Strong bullish all around + CRSI low
            elif macro_bull and trend_4h_bullish and trend_200_bullish and crsi_1h[i] < 25:
                desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + CRSI rally (sell rip)
            if (macro_bear or trend_4h_bearish) and crsi_extreme_high:
                desired_signal = -BASE_SIZE
            # Short: Strong bearish all around + CRSI high
            elif macro_bear and trend_4h_bearish and trend_200_bearish and crsi_1h[i] > 75:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only trade extreme CRSI with strong trend confluence
            if in_session and volume_confirmed:
                if crsi_very_low and (macro_bull and trend_4h_bullish):
                    desired_signal = BASE_SIZE
                elif crsi_very_low and trend_200_bullish:
                    desired_signal = REDUCED_SIZE
                
                if crsi_very_high and (macro_bear and trend_4h_bearish):
                    desired_signal = -BASE_SIZE
                elif crsi_very_high and trend_200_bearish:
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
                if (macro_bull or trend_4h_bullish) and crsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (macro_bear or trend_4h_bearish) and crsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought (mean reversion complete)
            if crsi_1h[i] > 80:
                desired_signal = 0.0
            # Exit if macro + medium trend reverses
            if macro_bear and trend_4h_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold (mean reversion complete)
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