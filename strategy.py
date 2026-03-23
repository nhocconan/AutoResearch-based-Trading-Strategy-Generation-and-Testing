#!/usr/bin/env python3
"""
Experiment #345: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Previous 1h failures (#335, #338, #340) had issues with:
1. Too many regime filters killing all trades (Sharpe=0.000 = 0 trades)
2. Trend-following doesn't work well on 1h for BTC/ETH (whipsaw)
3. Need mean reversion entries WITH HTF trend bias (not against it)

This strategy uses:
1. 1d HMA(21) as MACRO BIAS (hard filter: only long if price > 1d HMA)
2. 4h HMA(16) for intermediate trend confirmation
3. 1h Connors RSI for entry timing (CRSI < 15 long, CRSI > 85 short)
4. Session filter: only 8-20 UTC (major market hours, avoids Asia chop)
5. Volume confirmation: volume > 0.8x 20-bar average
6. Choppiness Index: CHOP > 55 favors mean reversion, CHOP < 45 favors trend
7. ATR trailing stop: 2.5x ATR from entry

KEY INSIGHT: Connors RSI has 75% win rate for mean reversion. Combined with
HTF trend bias, we only take mean reversion trades IN THE DIRECTION of the trend.
This reduces whipsaw while capturing pullbacks in trending markets.

TARGET: 40-80 trades/year on 1h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_regime_4h1d_session_v2"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA with less lag.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
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
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - short-term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
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
        if streak[i] > 0:
            # Up streak - bullish
            streak_rsi[i] = min(100, 50 + streak_abs[i] * 10)
        elif streak[i] < 0:
            # Down streak - bearish
            streak_rsi[i] = max(0, 50 - streak_abs[i] * 10)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current change ranks in lookback
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        changes = close_s.iloc[i-rank_period+1:i+1].diff().dropna().values
        if len(changes) > 0:
            current_change = close[i] - close[i-1]
            pct_rank[i] = 100 * np.sum(changes <= current_change) / len(changes)
        else:
            pct_rank[i] = 50
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return pd.to_datetime(open_time, unit='ms').hour

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
    
    # Volume moving average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (smaller due to more trades)
    
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # High choppiness = mean reversion favored
        is_trending = chop[i] < 45.0  # Low choppiness = trend favored
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_ma_20[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        crsi_neutral = 30.0 <= crsi[i] <= 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # LONG SETUP: 1d bullish + 4h bullish + CRSI oversold + volume
        if price_above_hma_1d and price_above_hma_4h:
            if crsi_oversold and volume_confirmed:
                # Strong long: all conditions aligned
                desired_signal = BASE_SIZE
            elif crsi[i] < 25 and is_choppy and volume_confirmed:
                # Mean reversion long in choppy market
                desired_signal = BASE_SIZE * 0.7
        
        # SHORT SETUP: 1d bearish + 4h bearish + CRSI overbought + volume
        elif price_below_hma_1d and price_below_hma_4h:
            if crsi_overbought and volume_confirmed:
                # Strong short: all conditions aligned
                desired_signal = -BASE_SIZE
            elif crsi[i] > 75 and is_choppy and volume_confirmed:
                # Mean reversion short in choppy market
                desired_signal = -BASE_SIZE * 0.7
        
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
        
        # === CRSI EXIT (opposite extreme) ===
        if in_position and position_side > 0 and crsi_overbought:
            # Long position, CRSI now overbought - take profit
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_oversold:
            # Short position, CRSI now oversold - take profit
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if trend bias still valid
            if position_side > 0 and price_above_hma_1d and price_above_hma_4h:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_1d and price_below_hma_4h:
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