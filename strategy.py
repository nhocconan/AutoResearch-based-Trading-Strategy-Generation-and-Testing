#!/usr/bin/env python3
"""
Experiment #820: 1h Primary + 4h/12h HTF — Connors RSI + Session + Volume Confluence

Hypothesis: After analyzing 553+ failed strategies, key insights for 1h timeframe:
1. 1h with 4h/12h HTF direction filter = HTF trade frequency with 1h entry precision
2. Connors RSI (RSI3 + StreakRSI + PercentRank) / 3 outperforms standard RSI for mean reversion
3. Session filter (8-20 UTC) eliminates 60% of low-volume whipsaw trades
4. Volume confirmation (0.8x avg) ensures real moves, not fake breakouts
5. Choppiness Index regime switch prevents trend-following in ranges
6. Discrete signals (0.0, ±0.25, ±0.35) minimize fee churn from signal changes
7. ATR trailing stop at 2.0x protects capital in 2022-style crashes

Strategy design:
1. 4h HMA(21) for intermediate trend direction (aligned via mtf_data)
2. 12h HMA(21) for long-term trend bias (aligned via mtf_data)
3. 1h Connors RSI for entry timing (extreme <10 long, >90 short)
4. 1h Choppiness Index(14) for regime detection (>55 range, <45 trend)
5. Session filter: only trade 8-20 UTC (high volume hours)
6. Volume filter: volume > 0.8x 20-period average
7. ATR(14) trailing stop at 2.0x for risk management
8. Discrete signals: 0.0, ±0.25, ±0.35 (max 0.40)

Key differences from #811 (4h primary):
- 1h primary allows more precise entry timing within HTF trend
- Connors RSI instead of standard RSI (better mean reversion signals)
- Session + Volume filters reduce trade count to target 40-60/year
- 4h + 12h dual HTF (not 1d + 1w) for better alignment with 1h entries

Target: Sharpe > 0.612, trades 40-80/year, ALL symbols positive Sharpe
Timeframe: 1h (with 4h/12h HTF direction filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_session_vol_chop_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, lookback=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Better for mean reversion than standard RSI.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < lookback + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_window > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100 if streak_period > 0 else 50
    
    # Percent Rank - where does current return rank vs last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(lookback, n):
        returns = np.diff(close[i-lookback:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = rank * 100
    
    # Combine into Connors RSI
    for i in range(lookback, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return np.clip(crsi, 0, 100)

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
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

def extract_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # open_time is in milliseconds
        hours[i] = (open_time_array[i] // 1000 // 3600) % 24
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, lookback=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for long-term trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Extract trading hours for session filter
    hours = extract_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.35
    REDUCED_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_avg_20[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === LONG-TERM TREND BIAS (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi_1h[i] < 10
        crsi_extreme_overbought = crsi_1h[i] > 90
        crsi_oversold = crsi_1h[i] < 25
        crsi_overbought = crsi_1h[i] > 75
        
        desired_signal = 0.0
        
        # Only trade during high-volume session hours
        if not in_session or not volume_ok:
            # Hold existing position but don't enter new
            if in_position:
                if position_side > 0 and (trend_4h_bullish or trend_12h_bullish):
                    desired_signal = BASE_SIZE
                elif position_side < 0 and (trend_4h_bearish or trend_12h_bearish):
                    desired_signal = -BASE_SIZE
            signals[i] = desired_signal
            continue
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI extreme oversold + any trend alignment
            if crsi_extreme_oversold and (trend_12h_bullish or trend_4h_bullish or above_sma200):
                desired_signal = BASE_SIZE
            elif crsi_oversold and (trend_12h_bullish or trend_4h_bullish):
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI extreme overbought + any trend alignment
            if crsi_extreme_overbought and (trend_12h_bearish or trend_4h_bearish or below_sma200):
                desired_signal = -BASE_SIZE
            elif crsi_overbought and (trend_12h_bearish or trend_4h_bearish):
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: 12h bullish + CRSI pullback (not extreme)
            if trend_12h_bullish and crsi_oversold and not crsi_extreme_oversold:
                desired_signal = BASE_SIZE
            elif trend_4h_bullish and crsi_oversold and above_sma200:
                desired_signal = REDUCED_SIZE
            
            # Short: 12h bearish + CRSI pullback
            if trend_12h_bearish and crsi_overbought and not crsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            elif trend_4h_bearish and crsi_overbought and below_sma200:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: only extreme CRSI with strong trend alignment
            if crsi_extreme_oversold and trend_12h_bullish and trend_4h_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_12h_bearish and trend_4h_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF trend intact and CRSI not overbought
                if (trend_12h_bullish or trend_4h_bullish) and crsi_1h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend intact and CRSI not oversold
                if (trend_12h_bearish or trend_4h_bearish) and crsi_1h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF trends reverse + CRSI overbought
            if trend_12h_bearish and trend_4h_bearish and crsi_1h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trends reverse + CRSI oversold
            if trend_12h_bullish and trend_4h_bullish and crsi_1h[i] < 20:
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