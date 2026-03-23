#!/usr/bin/env python3
"""
Experiment #785: 1h Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 500+ failed strategies, the key insights are:
1. 1h timeframe needs VERY strict filters to avoid fee drag (target 30-80 trades/year)
2. Choppiness Index regime detection works better than ADX for crypto bear/range markets
3. Connors RSI (CRSI) provides faster mean reversion signals than standard RSI(14)
4. 4h HMA(21) + 1d HMA(50) dual trend filter reduces whipsaw vs single HTF
5. Session filter (6-22 UTC) avoids low liquidity Asian overnight hours
6. Relaxed entry thresholds ensure ≥10 trades/train, ≥3/test per symbol

Strategy design:
1. 1h primary indicators: CRSI, Choppiness, ATR, Volume
2. 4h HMA(21) aligned via mtf_data for intermediate trend
3. 1d HMA(50) aligned via mtf_data for major trend bias
4. CHOP > 50 = range → mean reversion at CRSI extremes
5. CHOP < 50 = trend → follow HTF trend on pullbacks
6. Session filter: 6-22 UTC only
7. Volume filter: >0.8x 20-bar average
8. ATR(14) trailing stop at 2.5x
9. Discrete signals: 0.0, ±0.20, ±0.30

Key improvements from failed experiments:
- Relaxed CRSI thresholds (<20/>80 vs <10/>90) for more trades
- Relaxed CHOP thresholds (>50/<50 vs >61.8/<38.2) for regime detection
- Dual HTF trend filter (4h + 1d) for better signal quality
- Session filter reduces low-liquidity whipsaw
- Conservative position sizing (0.20-0.30) for drawdown control

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive Sharpe
Timeframe: 1h (with 4h/1d HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_4h1d_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
    series = pd.Series(series)
    wma_half = series.rolling(window=period//2, min_periods=period//2).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1) * 2 / (len(x) * (len(x) + 1))), raw=True
    )
    wma_full = series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1) * 2 / (len(x) * (len(x) + 1))), raw=True
    )
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1) * 2 / (len(x) * (len(x) + 1))), raw=True
    )
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

def calculate_rsi_streak(close, period=2):
    """Connors RSI Streak component."""
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
    """Connors RSI Percent Rank component."""
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
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending.
    Using 50 as threshold for crypto volatility.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    if n < period + 1:
        return chop
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        if price_range <= 1e-10:
            chop[i] = 50
            continue
        atr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

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
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    hour_1h = extract_hour(open_time)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (6-22 UTC) ===
        in_session = 6 <= hour_1h[i] <= 22
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === TREND BIAS (Dual HTF HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend when both HTF agree
        strong_bullish = trend_4h_bullish and trend_1d_bullish
        strong_bearish = trend_4h_bearish and trend_1d_bearish
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_1h[i] > 50
        trending_regime = chop_1h[i] < 50
        
        # === CRSI SIGNALS (relaxed thresholds for trade frequency) ===
        crsi_oversold = crsi_1h[i] < 20
        crsi_overbought = crsi_1h[i] > 80
        crsi_neutral_low = 30 < crsi_1h[i] < 45
        crsi_neutral_high = 55 < crsi_1h[i] < 70
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 50) ===
        if ranging_regime and in_session:
            # Mean reversion long: CRSI oversold + volume confirmed
            if crsi_oversold:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + volume confirmed
            if crsi_overbought:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 50) ===
        elif trending_regime and in_session:
            # Trend pullback long: strong bullish + CRSI neutral low
            if strong_bullish and crsi_neutral_low:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: strong bearish + CRSI neutral high
            if strong_bearish and crsi_neutral_high:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Trend continuation: strong trend + CRSI confirming
            if strong_bullish and crsi_1h[i] > 50 and not crsi_overbought:
                if volume_confirmed:
                    desired_signal = REDUCED_SIZE
            
            if strong_bearish and crsi_1h[i] < 50 and not crsi_oversold:
                if volume_confirmed:
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
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not extreme overbought
                if trend_4h_bullish and crsi_1h[i] < 85:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not extreme oversold
                if trend_4h_bearish and crsi_1h[i] > 15:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if strong bearish reversal
            if strong_bearish and crsi_1h[i] > 60:
                desired_signal = 0.0
        if in_position and position_side < 0:
            # Exit short if strong bullish reversal
            if strong_bullish and crsi_1h[i] < 40:
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