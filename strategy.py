#!/usr/bin/env python3
"""
Experiment #355: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Previous 1h strategies failed because:
1. Too many trades (>200/year) → fee drag kills profit on lower TF
2. Entry conditions not strict enough → whipsaw in choppy markets
3. No session filter → trading during low-liquidity hours (0-8 UTC)

This strategy uses PROVEN lower-TF pattern with EXTREME selectivity:
1. 1d HMA(21) as MACRO BIAS (hard filter: only long if price > 1d HMA)
2. 4h HMA(21) as INTERMEDIATE TREND (confirms 1d direction)
3. 1h Connors RSI for ENTRY TIMING (CRSI < 10 long, CRSI > 90 short)
4. Choppiness Index regime (CHOP > 55 = range/mean-revert, CHOP < 45 = trend)
5. SESSION FILTER: Only trade 8-20 UTC (high liquidity, avoid Asia overnight)
6. VOLUME FILTER: Volume > 0.8x 20-period average (avoid low-liquidity traps)
7. Position size: 0.20 (conservative for 1h TF)
8. ATR(14) trailing stop at 2.5x for risk management

KEY INSIGHT: 1h strategies MUST be extremely selective. Using 4h/1d for direction
and 1h only for entry timing gives HTF trade frequency with 1h execution precision.
Session + volume filters eliminate 60%+ of potential trades → target 30-60 trades/year.

TARGET: 30-60 trades/year on 1h, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_regime_4h1d_session_volume_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of today's price change over last 100 days
    
    Entry signals: CRSI < 10 (extreme oversold), CRSI > 90 (extreme overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[max(0, i-streak_period):i] > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100.0
    
    # Percent Rank - percentile of today's return over last 100 days
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[max(0, i-rank_period):i]
        if len(window) > 0:
            percent_rank[i] = (np.sum(window < returns[i]) / len(window)) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (breakout)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def extract_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) % 86400) // 3600
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
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume moving average for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract UTC hour for session filter
    utc_hours = extract_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # 20% position size for 1h (conservative, target 30-60 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        current_hour = utc_hours[i]
        in_session = (current_hour >= 8) and (current_hour <= 20)
        
        if not in_session:
            # Outside session hours - only maintain existing positions, don't enter new
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === VOLUME FILTER (> 0.8x 20-period average) ===
        volume_ok = volume[i] > (0.8 * vol_sma_20[i])
        
        if not volume_ok:
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA - CONFIRMATION) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # High choppiness = range regime (mean revert)
        is_trending = chop[i] < 45.0  # Low choppiness = trend regime (breakout)
        # Neutral zone (45-55): use existing position or stay flat
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        if price_above_hma_1d and price_above_hma_4h:
            # Bullish macro + intermediate trend
            
            if is_choppy:
                # RANGE REGIME: Mean reversion long on extreme oversold
                # CRSI < 10 = extreme oversold (Connors RSI signal)
                if crsi[i] < 10:
                    desired_signal = BASE_SIZE
            
            elif is_trending:
                # TREND REGIME: Pullback long in uptrend
                # CRSI < 25 = moderate pullback in trend
                if crsi[i] < 25:
                    desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        elif price_below_hma_1d and price_below_hma_4h:
            # Bearish macro + intermediate trend
            
            if is_choppy:
                # RANGE REGIME: Mean reversion short on extreme overbought
                # CRSI > 90 = extreme overbought
                if crsi[i] > 90:
                    desired_signal = -BASE_SIZE
            
            elif is_trending:
                # TREND REGIME: Pullback short in downtrend
                # CRSI > 75 = moderate rally in downtrend
                if crsi[i] > 75:
                    desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 80:
            # Long position: exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20:
            # Short position: exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if trend bias still valid
            if position_side > 0:
                if price_above_hma_1d and price_above_hma_4h:
                    # Bullish bias intact - hold long
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_1d and price_below_hma_4h:
                    # Bearish bias intact - hold short
                    desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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