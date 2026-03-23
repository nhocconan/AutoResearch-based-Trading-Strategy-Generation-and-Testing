#!/usr/bin/env python3
"""
Experiment #278: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion within Trend

Hypothesis: Lower TF (30m) strategies fail from too many trades → fee drag.
This version uses VERY STRICT confluence (3+ filters) to limit trades to 30-80/year:
- 4h HMA(16/48) for PRIMARY trend direction (mandatory)
- 1d HMA(21) for MACRO bias (soft filter)
- 30m Connors RSI for entry timing (extreme oversold/overbought only)
- Choppiness Index regime filter (CHOP > 55 = range → mean revert, CHOP < 45 = trend)
- Session filter (8-20 UTC only — highest liquidity)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.20 (conservative for 30m volatility)

Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
Long: CRSI < 15 (extreme oversold) + 4h bullish trend
Short: CRSI > 85 (extreme overbought) + 4h bearish trend

KEY: Only enter at CRSI extremes (<15 or >85) within HTF trend direction.
This should generate 30-80 trades/year on 30m, not 200+.

TARGET: Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL), trades 30-80/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_hma_4h1d_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days (streak length)
    PercentRank: percentile rank of price change over last 100 bars
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI of Streak (2) - measure consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100) - percentile of current change vs last 100 bars
    pct_rank = np.zeros(n)
    for i in range(100, n):
        changes = close_s.iloc[i-99:i+1].diff().iloc[1:]  # last 100 changes
        current_change = delta.iloc[i]
        if len(changes) > 0:
            pct_rank[i] = 100.0 * (changes < current_change).sum() / len(changes)
        else:
            pct_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak + pct_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy (mean reversion favored)
    CHOP < 38.2 = trending (trend following favored)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (highest - lowest + 1e-10)) / np.log10(period)
    
    return np.nan_to_num(chop, nan=50.0)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_30m = calculate_crsi(close)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, 48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.20  # Conservative for 30m (smaller than 1h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi_30m[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = (hour_utc >= 8) and (hour_utc <= 20)
        
        if not in_session:
            # If in position, hold; if not, don't enter
            if in_position:
                signals[i] = POSITION_SIZE if position_side > 0 else -POSITION_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) - SOFT FILTER ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) - PRIMARY FILTER ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === CHOPPING INDEX REGIME ===
        chop_range = chop_14[i] > 55.0  # Range market → mean reversion
        chop_trend = chop_14[i] < 45.0  # Trending market → trend follow
        
        # === CONNORS RSI EXTREMES (very strict) ===
        crsi_oversold = crsi_30m[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi_30m[i] > 85.0  # Extreme overbought
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bullish + CRSI oversold + (range OR trend regime) + session
        # Only enter mean reversion in range, trend follow in trend
        if hma_4h_bullish:
            if chop_range and crsi_oversold:
                # Range market: mean reversion long at extreme oversold
                desired_signal = POSITION_SIZE
            elif chop_trend and crsi_oversold and price_above_hma_1d:
                # Trend market: pullback long with macro bias
                desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 4h bearish + CRSI overbought + (range OR trend regime) + session
        if hma_4h_bearish:
            if chop_range and crsi_overbought:
                # Range market: mean reversion short at extreme overbought
                desired_signal = -POSITION_SIZE
            elif chop_trend and crsi_overbought and price_below_hma_1d:
                # Trend market: pullback short with macro bias
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit at opposite extreme) ===
        if in_position and position_side > 0 and crsi_30m[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_30m[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_4h_bullish:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and hma_4h_bearish:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals