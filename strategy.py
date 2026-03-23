#!/usr/bin/env python3
"""
Experiment #825: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After 563+ failed strategies, the key insight is that 1h timeframe needs
HTF trend filtering to reduce trade frequency (target 30-60 trades/year) while using
1h only for precise entry timing. Pure 1h strategies generate too many trades → fee drag.

Strategy design:
1. 1h Primary timeframe for entry timing
2. 4h HMA(21) for trend direction (long only when 4h HMA bullish)
3. 1d Choppiness Index(14) for regime detection (CHOP>55=range, CHOP<45=trend)
4. 1h Connors RSI for entry (CRSI<15 long, CRSI>85 short)
5. Session filter: only trade 8-20 UTC (high volume hours)
6. Volume filter: current volume > 0.8x 20-period avg
7. ATR(14) trailing stop 2.5x for risk management
8. Position size: 0.25 (conservative for 1h TF)

Why this should work:
- 4h HMA filters out counter-trend trades (major improvement over pure 1h)
- Connors RSI has proven 75% win rate for mean reversion
- Session filter avoids low-volume whipsaws (Asian session)
- CHOP regime adapts between mean-revert and trend-follow
- Discrete signal levels (0.0, ±0.25) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-60 trades/year with HTF filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_hma_4h1d_atr_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    Long: CRSI < 10-15, Short: CRSI > 85-90
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period + 1:
        return crsi
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    delta = np.diff(close)
    
    for i in range(1, n):
        streak = 0
        if delta[i-1] > 0:
            # Count consecutive positive days
            j = i - 1
            while j >= 0 and delta[j] > 0:
                streak += 1
                j -= 1
            streak_rsi[i] = 100 * streak / streak_period if streak_period > 0 else 0
        elif delta[i-1] < 0:
            # Count consecutive negative days
            j = i - 1
            while j >= 0 and delta[j] < 0:
                streak += 1
                j -= 1
            streak_rsi[i] = 100 * (1 - streak / streak_period) if streak_period > 0 else 100
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component (where does current return rank vs last 100?)
    pr = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(pr_period, n):
        window = returns[i-pr_period+1:i+1]
        current = returns[i]
        count_below = np.sum(window[:-1] < current)  # exclude current from comparison
        pr[i] = 100 * count_below / (pr_period - 1)
    
    # Combine all three components
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

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
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
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
        prev_close = close[i-1]
        tr[i] = max(high[i] - low[i], np.abs(high[i] - prev_close), np.abs(low[i] - prev_close))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
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
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_20[i]):
            continue
        
        # Extract UTC hour for session filter
        hour_utc = get_hour_from_timestamp(open_time[i])
        
        # === SESSION FILTER (8-20 UTC = high volume hours) ===
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER (current > 0.8x 20-period avg) ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === TREND DIRECTION (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d_aligned[i] > 55
        trending_regime = chop_1d_aligned[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 15
        crsi_overbought = crsi_1h[i] > 85
        crsi_extreme_oversold = crsi_1h[i] < 10
        crsi_extreme_overbought = crsi_1h[i] > 90
        crsi_recovering = crsi_1h[i] > crsi_1h[i-1] and crsi_1h[i] < 30
        crsi_weakening = crsi_1h[i] < crsi_1h[i-1] and crsi_1h[i] > 70
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime and in_session and volume_ok:
            # Long: CRSI oversold + 4h trend neutral or bullish
            if crsi_oversold and (trend_4h_bullish or not trend_4h_bearish):
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + 4h trend neutral or bearish
            if crsi_overbought and (trend_4h_bearish or not trend_4h_bullish):
                desired_signal = -BASE_SIZE
            
            # Extreme CRSI alone (guarantees trades)
            if crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime and in_session and volume_ok:
            # Long: 4h bullish + CRSI recovering from oversold
            if trend_4h_bullish and crsi_recovering:
                desired_signal = BASE_SIZE
            
            # Short: 4h bearish + CRSI weakening from overbought
            if trend_4h_bearish and crsi_weakening:
                desired_signal = -BASE_SIZE
            
            # Pullback entry in trend
            if trend_4h_bullish and crsi_oversold:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if trend_4h_bearish and crsi_overbought:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            if in_session and volume_ok:
                # Conservative: extreme CRSI + trend alignment
                if crsi_extreme_oversold and trend_4h_bullish:
                    desired_signal = REDUCED_SIZE
                
                if crsi_extreme_overbought and trend_4h_bearish:
                    desired_signal = -REDUCED_SIZE
                
                # Basic mean reversion
                if crsi_oversold and not trend_4h_bearish:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
                
                if crsi_overbought and not trend_4h_bullish:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
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
                if trend_4h_bullish and crsi_1h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and CRSI not oversold
                if trend_4h_bearish and crsi_1h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses + CRSI overbought
            if trend_4h_bearish and crsi_1h[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses + CRSI oversold
            if trend_4h_bullish and crsi_1h[i] < 15:
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