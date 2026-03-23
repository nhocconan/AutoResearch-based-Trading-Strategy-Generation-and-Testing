#!/usr/bin/env python3
"""
Experiment #675: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness + Volume + Session

Hypothesis: 1h timeframe with strict confluence filters can achieve optimal trade frequency
(30-80/year) while maintaining high win rate. Key innovations:
1. Connors RSI (CRSI) — proven 75% win rate for mean reversion entries
2. 4h HMA for trend direction — don't fight the higher timeframe trend
3. 1d Choppiness Index regime — CHOP>55=range(mean-revert), CHOP<45=trend
4. Volume confirmation — volume > 0.8x 20-period average (avoids low-liquidity traps)
5. Session filter — only 8-20 UTC (high liquidity, reduces noise)
6. Asymmetric sizing — 0.25 long, 0.20 short (crypto long bias)
7. 2.5x ATR trailing stop — protects profits while allowing room

Why this should work where 1h strategies failed:
- Exp #665, #670 failed with 0 trades or negative Sharpe on 1h
- Too many filters = 0 trades, too few = fee drag
- CRSI extremes (10/90) happen ~5-10% of time = ~40-80 signals/year on 1h
- HTF trend filter removes ~50% of counter-trend signals
- Session filter removes ~40% of low-liquidity hours
- Combined: ~15-30 trades/year = optimal for 1h TF

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_volume_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — composite momentum indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long signal: CRSI < 10 (extreme oversold)
    Short signal: CRSI > 90 (extreme overbought)
    Proven 75% win rate in mean-reversion strategies.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    rsi_close = np.concatenate([[np.nan], rsi_close])
    
    # RSI(streak, 2) — streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).rolling(window=streak_period, min_periods=streak_period).mean().values
    avg_streak_loss = pd.Series(streak_loss).rolling(window=streak_period, min_periods=streak_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # PercentRank(100) — where current close ranks in last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — identifies ranging vs trending markets.
    CHOP > 61.8 = choppy/ranging (mean-revert)
    CHOP < 38.2 = trending (trend-follow)
    We use 55/45 thresholds for more signals.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr1 = high[j] - low[j]
                tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
                tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
                atr_sum += max(tr1, tr2, tr3)
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs 20-period average."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    if n < period:
        return vol_ratio
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_ma + 1e-10)
    
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000
    utc_hour = pd.to_datetime(ts_seconds, unit='s').hour
    return utc_hour

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
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF (4h) indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF (1d) indicators
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after warmup period (CRSI needs 100 + buffer)
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ratio_1h[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === REGIME DETECTION (Daily Choppiness) ===
        chop_daily = chop_1d_aligned[i]
        is_range_regime = chop_daily > 55  # Mean-revert in choppy markets
        is_trend_regime = chop_daily < 45  # Trend-follow in trending markets
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === 1D TREND BIAS (stronger filter) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi_1h[i] < 12  # Extreme oversold
        crsi_overbought = crsi_1h[i] > 88  # Extreme overbought
        crsi_moderate_oversold = crsi_1h[i] < 25
        crsi_moderate_overbought = crsi_1h[i] > 75
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio_1h[i] > 0.8  # At least 80% of avg volume
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Need: session + volume + (CRSI extreme OR CRSI moderate + trend align)
        if in_session and volume_confirmed:
            # Strong long: CRSI extreme oversold + 4h trend bullish (or range regime)
            if crsi_oversold:
                if trend_4h_bullish or is_range_regime:
                    desired_signal = SIZE_LONG
                # In strong 1d uptrend, even counter-4h is OK for mean reversion
                elif trend_1d_bullish and crsi_1h[i] < 8:
                    desired_signal = SIZE_LONG * 0.5
            
            # Moderate long: CRSI moderate + strong trend alignment
            elif crsi_moderate_oversold:
                if trend_4h_bullish and trend_1d_bullish:
                    desired_signal = SIZE_LONG
                elif is_range_regime and trend_4h_bullish:
                    desired_signal = SIZE_LONG
        
        # === SHORT ENTRY CONDITIONS ===
        if in_session and volume_confirmed:
            # Strong short: CRSI extreme overbought + 4h trend bearish (or range regime)
            if crsi_overbought:
                if trend_4h_bearish or is_range_regime:
                    desired_signal = -SIZE_SHORT
                # In strong 1d downtrend, even counter-4h is OK for mean reversion
                elif trend_1d_bearish and crsi_1h[i] > 92:
                    desired_signal = -SIZE_SHORT * 0.5
            
            # Moderate short: CRSI moderate + strong trend alignment
            elif crsi_moderate_overbought:
                if trend_4h_bearish and trend_1d_bearish:
                    desired_signal = -SIZE_SHORT
                elif is_range_regime and trend_4h_bearish:
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not extremely overbought AND 4h trend still bullish
                if crsi_1h[i] < 80 and trend_4h_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if CRSI not extremely oversold AND 4h trend still bearish
                if crsi_1h[i] > 20 and trend_4h_bearish:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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