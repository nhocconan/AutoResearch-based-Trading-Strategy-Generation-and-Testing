#!/usr/bin/env python3
"""
Experiment #168: 30m Primary + 4h/1d HTF — Regime-Adaptive CRSI with Session Filter

Hypothesis: Lower TF (30m) strategies fail due to fee drag from too many trades.
This strategy uses HTF (4h/1d) for SIGNAL DIRECTION and regime detection,
30m ONLY for entry timing within the HTF trend structure.

Key innovations vs failed strategies:
1) 4h HMA(21) = primary trend direction (ONLY trade WITH 4h trend)
2) 1d Choppiness Index = regime filter (CHOP>55 range, CHOP<45 trend)
3) 30m Connors RSI = entry timing (CRSI<15 long, CRSI>85 short)
4) Session filter = only 8-20 UTC (avoid low-liquidity whipsaw)
5) Volume confirmation = volume > 0.8x 20-bar avg
6) Discrete position sizing = 0.0, ±0.20, ±0.30 (minimize fee churn)

Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
Position size: 0.20 base, 0.30 with full HTF confluence
Stoploss: 2.0x ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_hma_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    chop = chop.fillna(50.0).values
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback period
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_fast = 100.0 - (100.0 / (1.0 + rs))
    rsi_fast = rsi_fast.fillna(50.0)
    
    # RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0)
    
    # Percent Rank (percentile of price change over lookback)
    price_change = close_s.diff()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = price_change.iloc[i-rank_period:i]
        if len(window) > 0:
            rank = (window < price_change.iloc[i]).sum()
            percent_rank.iloc[i] = rank / len(window) * 100.0
    
    percent_rank = percent_rank.fillna(50.0)
    
    # Connors RSI
    crsi = (rsi_fast + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

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
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d Choppiness Index for regime filter
    chop_1d_raw = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_1d_aligned[i]) or np.isnan(crsi_30m[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour (Binance timestamps are in milliseconds)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24
        session_ok = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === REGIME FILTER (1d Choppiness) ===
        # CHOP > 55 = range market (favor mean reversion)
        # CHOP < 45 = trending market (favor trend following)
        regime_range = chop_1d_aligned[i] > 55.0
        regime_trend = chop_1d_aligned[i] < 45.0
        
        # === HTF TREND DIRECTION (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 15 = oversold (long signal)
        # CRSI > 85 = overbought (short signal)
        crsi_oversold = crsi_30m[i] < 15.0
        crsi_overbought = crsi_30m[i] > 85.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Range regime: price above 4h HMA (bullish bias) + CRSI oversold
        if regime_range and price_above_hma_4h and crsi_oversold and session_ok and volume_ok:
            new_signal = POSITION_SIZE_BASE
        
        # Trend regime: price above 4h HMA + CRSI moderately oversold (< 30)
        if regime_trend and price_above_hma_4h and crsi_30m[i] < 30.0 and session_ok and volume_ok:
            new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Range regime: price below 4h HMA (bearish bias) + CRSI overbought
        if regime_range and price_below_hma_4h and crsi_overbought and session_ok and volume_ok:
            new_signal = -POSITION_SIZE_BASE
        
        # Trend regime: price below 4h HMA + CRSI moderately overbought (> 70)
        if regime_trend and price_below_hma_4h and crsi_30m[i] > 70.0 and session_ok and volume_ok:
            new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid (no exit signal yet)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 4h HMA
                if price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 4h HMA
                if price_below_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 4h HMA
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        # Exit short if price crosses above 4h HMA
        if in_position and position_side < 0 and price_above_hma_4h:
            new_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit long if CRSI > 70 (overbought)
        if in_position and position_side > 0 and crsi_30m[i] > 70.0:
            new_signal = 0.0
        
        # Exit short if CRSI < 30 (oversold)
        if in_position and position_side < 0 and crsi_30m[i] < 30.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals