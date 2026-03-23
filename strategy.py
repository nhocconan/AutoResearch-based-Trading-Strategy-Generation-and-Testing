#!/usr/bin/env python3
"""
Experiment #285: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Previous 1h strategies failed from over-filtering (0 trades) or simple trend following
(negative Sharpe in bear markets). This version uses:
- 4h HMA(21) for MACRO trend direction (soft bias, not hard filter)
- 1h Connors RSI for mean reversion entries (CRSI < 20 long, > 80 short)
- Choppiness Index to adjust position size (range = full size, trend = half size)
- 1d HMA for ultra-long-term bias (only avoid counter-trend trades)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.20-0.30 (conservative for 1h, adjusted by regime)

KEY INSIGHT: Mean reversion works better in bear/range markets (2022, 2025) than trend following.
Connors RSI has 75% win rate on pullbacks. Choppiness Index tells us WHEN to mean revert vs trend.

TARGET: 40-80 trades/year on 1h, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h1d_atr_v1"
timeframe = "1h"
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
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measure consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    # Convert streak to RSI-like value (0-100)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent Rank - where current return ranks vs last N periods
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period+1:i+1].dropna()
        if len(window) > 0:
            current = returns.iloc[i]
            if not np.isnan(current):
                percent_rank[i] = (window < current).sum() / len(window) * 100.0
            else:
                percent_rank[i] = 50.0
        else:
            percent_rank[i] = 50.0
    percent_rank[:rank_period] = 50.0
    
    # CRSI = average of 3 components
    crsi = (rsi_fast + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR(1), n) / (Highest High(n) - Lowest Low(n))) / LOG10(n)
    
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # ATR(1) = true range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    tr_s = pd.Series(tr)
    
    # Sum of ATR(1) over period
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    hhll = hh - ll
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (hhll + 1e-10)) / np.log10(period)
    
    chop = chop.fillna(50.0).values
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size for 1h
    MAX_SIZE = 0.35   # Max size in strong range regime
    MIN_SIZE = 0.15   # Min size in trend regime
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Range market - mean revert
        is_trend = chop[i] < 45.0  # Trend market - follow trend
        
        # Position size adjustment based on regime
        if is_range:
            position_size = MAX_SIZE  # Full size in range (mean reversion works best)
        elif is_trend:
            position_size = MIN_SIZE  # Half size in trend (mean reversion risky)
        else:
            position_size = BASE_SIZE  # Normal size in transition
        
        # === MACRO BIAS (1d HMA) - SOFT FILTER ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA) - DIRECTION BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CONNORS RSI SIGNALS (mean reversion extremes) ===
        crsi_oversold = crsi[i] < 20.0  # Strong buy signal
        crsi_overbought = crsi[i] > 80.0  # Strong sell signal
        crsi_neutral = (crsi[i] >= 20.0) and (crsi[i] <= 80.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + 4h not strongly bearish (avoid counter-trend)
        # In range market: enter on any CRSI < 20
        # In trend market: only enter if aligned with 4h trend
        if crsi_oversold:
            if is_range:
                # Range market - mean revert freely
                desired_signal = position_size
            elif is_trend and price_above_hma_4h:
                # Trend market - only long if 4h bullish
                desired_signal = position_size
            elif not is_trend and not price_below_hma_4h:
                # Transition - avoid strong counter-trend
                desired_signal = position_size * 0.5
        
        # SHORT ENTRY: CRSI overbought + 4h not strongly bullish
        elif crsi_overbought:
            if is_range:
                # Range market - mean revert freely
                desired_signal = -position_size
            elif is_trend and price_below_hma_4h:
                # Trend market - only short if 4h bearish
                desired_signal = -position_size
            elif not is_trend and not price_above_hma_4h:
                # Transition - avoid strong counter-trend
                desired_signal = -position_size * 0.5
        
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
        
        # === CRSI MEAN REVERSION EXIT (take profit) ===
        # Exit long when CRSI goes overbought, exit short when CRSI goes oversold
        if in_position and position_side > 0 and crsi_overbought:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_oversold:
            desired_signal = 0.0
        
        # === 4h TREND REVERSAL EXIT ===
        # Exit if 4h trend strongly reverses against position
        if in_position and position_side > 0 and price_below_hma_4h and chop[i] < 40.0:
            # Strong trend reversal against long
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_4h and chop[i] < 40.0:
            # Strong trend reversal against short
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        # If in position and no exit signal, maintain position
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and not crsi_overbought:
                desired_signal = position_side * position_size
            elif position_side < 0 and not crsi_oversold:
                desired_signal = position_side * position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    highest_since_entry = 0.0
                    lowest_since_entry = close[i]
            elif np.sign(desired_signal) != position_side:
                # Position reversal
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    highest_since_entry = 0.0
                    lowest_since_entry = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals