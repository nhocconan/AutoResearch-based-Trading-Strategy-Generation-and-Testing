#!/usr/bin/env python3
"""
Experiment #765: 1h Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 500+ failed experiments, the key insight is:
1. 1h timeframe needs VERY strict filters to avoid fee drag (target 30-80 trades/year)
2. Choppiness Index (CHOP) is superior to ADX for crypto regime detection
3. Connors RSI (CRSI) provides faster mean reversion signals than standard RSI
4. 4h HMA(21) gives smoother trend bias than EMA for 1h entries
5. 1d EMA(50) provides ultimate trend filter to avoid counter-trend trades
6. Session filter (8-20 UTC) captures high-liquidity periods only
7. Volume filter ensures we trade only on meaningful moves

Strategy design:
1. 1d EMA(50) — ultimate trend bias (aligned via mtf_data)
2. 4h HMA(21) — intermediate trend filter (aligned via mtf_data)
3. 1h Choppiness Index(14) — regime detection (>55=range, <45=trend)
4. 1h Connors RSI — entry timing (RSI(3)+Streak(2)+PercentRank(100))/3
5. 1h Bollinger Bands(20,2.0) — mean reversion bounds
6. 1h Volume — confirm with >1.0x 20-bar average
7. Session filter — only trade 8-20 UTC (high liquidity)
8. ATR(14) trailing stop — 2.5x for risk management

Key differences from failed 1h strategies (#755, #758, #760):
- Looser CRSI thresholds (<20/>80 vs <10/>90) to ensure trades generate
- Removed conflicting filters that caused 0 trades
- Added hold logic to maintain positions through minor pullbacks
- Session filter reduces noise but doesn't block all entries
- Volume filter relaxed to 1.0x (not 1.5x) for more trade frequency

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year with strict filters)
Position size: 0.20-0.30 (conservative for 1h timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_hma(series, period):
    """
    Hull Moving Average — smoother and more responsive than EMA.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = pd.Series(series).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean().values
    
    return hma

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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

def calculate_rsi_streak(close, period=2):
    """
    Connors RSI Streak component.
    Measures consecutive up/down bars.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank component.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <20 = oversold, >80 = overbought (looser than standard).
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

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
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    atr_vals = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.nansum(atr_vals[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

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
    utc_hours = get_utc_hour(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF EMA for ultimate trend filter (1d)
    ema_1d_raw = calculate_ema(df_1d['close'].values, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
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
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(bb_sma[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(chop_1h[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === TREND BIAS (4h HMA21 + 1d EMA50) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > ema_1d_aligned[i]
        trend_1d_bearish = close[i] < ema_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        # Neutral regime: 45 <= CHOP <= 55
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.0 * vol_sma_1h[i]
        
        # === CRSI SIGNALS (looser thresholds for trade frequency) ===
        crsi_oversold = crsi_1h[i] < 20
        crsi_overbought = crsi_1h[i] > 80
        crsi_extreme_oversold = crsi_1h[i] < 15
        crsi_extreme_overbought = crsi_1h[i] > 85
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + below BB lower
            if crsi_oversold and below_bb_lower:
                if trend_4h_bullish or not trend_1d_bearish:
                    desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + above BB upper
            if crsi_overbought and above_bb_upper:
                if trend_4h_bearish or not trend_1d_bullish:
                    desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) ===
        elif trending_regime:
            # Trend pullback long: 4h bullish + 1d bullish + CRSI pullback
            if trend_4h_bullish and trend_1d_bullish:
                if 25 < crsi_1h[i] < 45:
                    desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: 4h bearish + 1d bearish + CRSI pullback
            if trend_4h_bearish and trend_1d_bearish:
                if 55 < crsi_1h[i] < 75:
                    desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: only extreme CRSI + trend alignment
            if crsi_extreme_oversold and trend_4h_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_4h_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === SESSION FILTER APPLICATION ===
        # Only allow new entries during session, but hold positions through off-hours
        if not in_session and desired_signal != 0.0 and not in_position:
            desired_signal = 0.0
        
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
                # Hold long if 4h trend intact and CRSI not overbought
                if trend_4h_bullish and crsi_1h[i] < 85:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend intact and CRSI not oversold
                if trend_4h_bearish and crsi_1h[i] > 15:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses or CRSI extremely overbought
            if trend_4h_bearish and crsi_1h[i] > 75:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper and crsi_1h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses or CRSI extremely oversold
            if trend_4h_bullish and crsi_1h[i] < 25:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower and crsi_1h[i] < 30:
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