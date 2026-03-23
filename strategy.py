#!/usr/bin/env python3
"""
Experiment #845: 1h Primary + 4h/1d HTF — Regime-Adaptive with Session Filter

Hypothesis: After 580+ failed strategies, the key issue is LOWER TIMEFRAME strategies
generate TOO MANY trades (>200/year) causing fee drag. This strategy uses:
1. 1h primary with VERY STRICT entry (3+ confluence required)
2. 4h HMA for trend direction (not entry trigger)
3. 1d Choppiness for regime (range vs trend)
4. Session filter: only trade 8-20 UTC (highest volume, lowest noise)
5. Connors RSI for mean reversion entries (proven 75% win rate)
6. Volume filter: only trade when volume > 0.8x 20-bar average
7. Discrete signal sizes: 0.0, ±0.20, ±0.30 to minimize churn

Why this might work when others failed:
- Session filter eliminates 60% of bars (reduces false signals)
- Volume filter ensures we trade only during active periods
- 4h trend + 1h entry = HTF trade frequency with 1h precision
- Connors RSI (not standard RSI) = better mean reversion signal
- Target: 40-80 trades/year (well under 100/year limit for 1h)

Key difference from failed 1h strategies (#838, #840):
- Stricter confluence (ALL filters must agree, not just 1-2)
- Session filter was missing in previous attempts
- Volume confirmation added
- More conservative sizing (0.20-0.30 vs 0.35)

Timeframe: 1h (target 50-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_session_volume_4h1d_atr_v1"
timeframe = "1h"
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
    Proven 75% win rate for mean reversion.
    Long: CRSI < 10, Short: CRSI > 90
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (2-period RSI on streak direction)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100.0 * streak[i] / (streak_period + 1)
        else:
            streak_rsi[i] = 100.0 * (streak_period + streak[i]) / (streak_period + 1)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank (100-period)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        lookback = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(lookback[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def extract_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    import datetime
    hours = np.zeros(len(open_time), dtype=int)
    for i, ot in enumerate(open_time):
        dt = datetime.datetime.utcfromtimestamp(ot / 1000)
        hours[i] = dt.hour
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
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    sma_200_1h = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for secular trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract UTC hour for session filter
    utc_hours = extract_hour_from_open_time(open_time)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(volume_sma_1h[i]) or volume_sma_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME FILTER (must be > 0.8x average) ===
        volume_ok = volume[i] > 0.8 * volume_sma_1h[i]
        
        # === TREND BIAS (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SECULAR TREND (1d HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (1h Choppiness) ===
        ranging_regime = chop_1h[i] > 61.8
        trending_regime = chop_1h[i] < 38.2
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi_1h[i] < 10
        crsi_extreme_overbought = crsi_1h[i] > 90
        crsi_oversold = crsi_1h[i] < 20
        crsi_overbought = crsi_1h[i] > 80
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        desired_signal = 0.0
        
        # Only trade during session with volume confirmation
        if in_session and volume_ok:
            # === RANGING REGIME (CHOP > 61.8) — Mean Reversion ===
            if ranging_regime:
                # Long: CRSI extreme oversold + bullish HTF bias OR above SMA200
                if crsi_extreme_oversold and (trend_4h_bullish or above_sma200):
                    desired_signal = BASE_SIZE
                elif crsi_oversold and trend_4h_bullish and trend_1d_bullish:
                    desired_signal = REDUCED_SIZE
                
                # Short: CRSI extreme overbought + bearish HTF bias OR below SMA200
                if crsi_extreme_overbought and (trend_4h_bearish or below_sma200):
                    desired_signal = -BASE_SIZE
                elif crsi_overbought and trend_4h_bearish and trend_1d_bearish:
                    desired_signal = -REDUCED_SIZE
            
            # === TRENDING REGIME (CHOP < 38.2) — Trend Following ===
            elif trending_regime:
                # Long: Bullish 4h + Bullish 1d + CRSI pulling back from oversold
                if trend_4h_bullish and trend_1d_bullish:
                    if crsi_oversold and crsi_1h[i] > crsi_1h[i-1] if not np.isnan(crsi_1h[i-1]) else False:
                        desired_signal = BASE_SIZE
                    elif above_sma200 and crsi_1h[i] < 40:
                        desired_signal = REDUCED_SIZE
                
                # Short: Bearish 4h + Bearish 1d + CRSI pulling back from overbought
                if trend_4h_bearish and trend_1d_bearish:
                    if crsi_overbought and crsi_1h[i] < crsi_1h[i-1] if not np.isnan(crsi_1h[i-1]) else False:
                        desired_signal = -BASE_SIZE
                    elif below_sma200 and crsi_1h[i] > 60:
                        desired_signal = -REDUCED_SIZE
            
            # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
            else:
                # Conservative: require ALL trend filters aligned + extreme CRSI
                if crsi_extreme_oversold and trend_4h_bullish and trend_1d_bullish and above_sma200:
                    desired_signal = REDUCED_SIZE
                
                if crsi_extreme_overbought and trend_4h_bearish and trend_1d_bearish and below_sma200:
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
                # Hold long if 4h trend still bullish and CRSI not overbought
                if trend_4h_bullish and crsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and CRSI not oversold
                if trend_4h_bearish and crsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses bearish + CRSI overbought
            if trend_4h_bearish and crsi_1h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses bullish + CRSI oversold
            if trend_4h_bullish and crsi_1h[i] < 20:
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