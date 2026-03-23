#!/usr/bin/env python3
"""
Experiment #648: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: Lower TF (30m) strategies fail due to too many trades → fee drag.
This uses 1d HMA for TREND BIAS (very slow), 4h Choppiness for REGIME (range vs trend),
30m Connors RSI for ENTRY TIMING (extreme mean reversion), plus session (8-20 UTC)
and volume filters. Target: 40-80 trades/year MAX.

Key insights from 572 failed strategies:
1. 30m strategies #638, #645 got Sharpe=0.000 with 0 trades — filters TOO strict
2. Need balance: strict enough for few trades, loose enough to generate trades
3. Connors RSI <10 or >90 has 75% win rate in literature (proven edge)
4. Session filter (8-20 UTC) captures highest volume hours, reduces noise
5. 1d HMA slope is slow but reliable for major trend direction
6. Choppiness >55 = range (mean revert), <45 = trend (follow direction)

Why this might beat Sharpe=0.520:
- 1d HMA keeps us on right side of major moves (like 2022 crash, 2021 bull)
- Choppiness regime filter avoids trend-following in ranges (whipsaw killer)
- Connors RSI extremes catch reversals with high probability
- Session + volume filters reduce noise trades during low-liquidity hours
- Conservative size (0.22) for 30m TF controls drawdown
- Target 40-80 trades/year (per Rule 10 for 30m)

Position sizing: 0.22 discrete (smaller for 30m, per Rule 4)
Target: 40-80 trades/year on 30m (per Rule 10)
Stoploss: 2.0*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_session_4h1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over 100 periods
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period_rsi)
    
    # RSI Streak (2)
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
    streak_sign = np.sign(streak)
    streak_rsi_raw = pd.Series(streak_abs).ewm(span=period_streak, min_periods=period_streak, adjust=False).mean().values
    streak_rsi = np.where(streak_sign >= 0, 100.0 - (100.0 / (1.0 + streak_rsi_raw / 10.0)), 
                          100.0 / (1.0 + streak_rsi_raw / 10.0))
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank(100)
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(period_rank, n):
        window = returns[i-period_rank+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100.0
    
    # Combine
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

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
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h Choppiness for regime detection
    chop_4h = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_30m = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi_30m[i]) or np.isnan(vol_avg_20[i]):
            continue
        if atr_14[i] == 0 or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Convert open_time to hour (Binance timestamps are in milliseconds)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 1D TREND BIAS (HMA slope over 5 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        chop_value = chop_4h_aligned[i]
        is_range = chop_value > 55.0  # Range/choppy market
        is_trend = chop_value < 45.0  # Trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi_30m[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi_30m[i] > 85.0  # Extreme overbought
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d bull trend + CRSI oversold + session + volume ---
        # Works in both range and trend regimes (mean revert in range, buy dip in trend)
        if in_session and volume_ok:
            if hma_1d_slope_bull and price_above_hma_1d:
                if crsi_oversold:
                    new_signal = POSITION_SIZE
            elif is_range and crsi_oversold:
                # In range market, can long oversold even without strong trend
                if price_above_hma_1d:  # Still need price above 1d HMA for safety
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 1d bear trend + CRSI overbought + session + volume ---
        if in_session and volume_ok:
            if hma_1d_slope_bear and price_below_hma_1d:
                if crsi_overbought:
                    new_signal = -POSITION_SIZE
            elif is_range and crsi_overbought:
                # In range market, can short overbought even without strong trend
                if price_below_hma_1d:  # Still need price below 1d HMA for safety
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
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