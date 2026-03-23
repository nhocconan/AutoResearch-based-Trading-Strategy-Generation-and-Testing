#!/usr/bin/env python3
"""
Experiment #293: 1d Primary + 1w HTF — Connors RSI + Donchian Regime Switch

Hypothesis: Recent regime strategies failed from over-complexity. This uses:
- Connors RSI (CRSI) for precise mean reversion entries (proven 75% win rate)
- 1w HMA(21) for MACRO trend bias (only trade CRSI signals in trend direction)
- Donchian(20) breakout confirmation for trend following mode
- Choppiness Index to switch between mean-revert and trend-follow regimes
- ATR(14) 3.0x trailing stoploss
- Position size: 0.28 (conservative for daily volatility)

KEY INSIGHT: CRSI works best when combined with HTF trend filter.
Long: CRSI<15 + price>1w_HMA + Donchian breakout confirmation
Short: CRSI>85 + price<1w_HMA + Donchian breakdown confirmation
Regime: CHOP>61.8 = mean revert at extremes, CHOP<38.2 = trend breakout

TARGET: 25-40 trades/year on 1d, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_donchian_regime_1w_hma_atr_v1"
timeframe = "1d"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * streak_abs[i] / max(streak_abs[max(0,i-streak_period+1):i+1].max(), 1)
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1 - streak_abs[i] / max(streak_abs[max(0,i-streak_period+1):i+1].max(), 1))
        else:
            streak_rsi[i] = 50.0
    
    # PercentRank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        current = close[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return np.clip(crsi, 0, 100)

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    choppiness = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for daily volatility
    
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === CRSI EXTREMES (Mean Reversion Signals) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT (Trend Following Signals) ===
        donchian_breakout_long = close[i] >= donchian_upper[i]
        donchian_breakdown_short = close[i] <= donchian_lower[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY - Mean Reversion Mode (ranging market)
        if is_ranging and crsi_oversold and price_above_hma_1w:
            desired_signal = POSITION_SIZE
        
        # LONG ENTRY - Trend Following Mode (trending market)
        elif is_trending and donchian_breakout_long and price_above_hma_1w:
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY - Mean Reversion Mode (ranging market)
        elif is_ranging and crsi_overbought and price_below_hma_1w:
            desired_signal = -POSITION_SIZE
        
        # SHORT ENTRY - Trend Following Mode (trending market)
        elif is_trending and donchian_breakdown_short and price_below_hma_1w:
            desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_1w:
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