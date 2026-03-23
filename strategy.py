#!/usr/bin/env python3
"""
Experiment #455: 1h Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion with Session Filter

Hypothesis: Lower timeframe (1h) strategies fail due to excessive trades → fee drag.
Solution: Use 4h/1d for TREND DIRECTION, 1h only for ENTRY TIMING within HTF trend.
Add session filter (8-20 UTC) to avoid low-liquidity periods. Use Choppiness Index
to switch between mean-reversion (range) and trend-follow regimes.

Key innovations:
1. 4h HMA(21) for primary trend bias (call ONCE before loop)
2. 1d HMA(21) for macro trend confirmation (call ONCE before loop)
3. Choppiness Index(14) for regime: >55=range(revert), <45=trend(follow)
4. Connors RSI(3,2,100) for precise entry timing in range regime
5. Session filter: only trade 8-20 UTC (high liquidity periods)
6. Volume confirmation: taker_buy_ratio >0.55 for longs, <0.45 for shorts
7. ATR(14) trailing stoploss at 2.5x
8. Position size: 0.20 base, 0.30 on strong confluence (discrete levels)

Target: Sharpe > 0.612, 40-80 trades/year, DD < -35%
Timeframe: 1h (lower TF requires stricter entry filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    n = len(close)
    
    rsi = calculate_rsi(close, rsi_period)
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_keltner(high, low, close, period=20, atr_period=10, mult=2.0):
    """Calculate Keltner Channel."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, lower, ema

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    kc_upper, kc_lower, kc_mid = calculate_keltner(high, low, close, period=20, atr_period=10, mult=2.0)
    
    # Calculate taker buy ratio for volume confirmation
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    # Calculate and align HTF HMA for bias (4h and 1d)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    # Calculate average volume for volume filter
    vol_sma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # 20% base position size for 1h (lower TF = smaller size)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop[i] > 55.0  # Range market → mean revert
        regime_trend = chop[i] < 45.0  # Trending market → trend follow
        
        # === HTF TREND BIAS (4h + 1d HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        htf_bullish = price_above_hma_4h and price_above_hma_1d
        htf_bearish = price_below_hma_4h and price_below_hma_1d
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === BOLLINGER BAND SIGNALS ===
        bb_break_lower = close[i] < bb_lower[i]
        bb_break_upper = close[i] > bb_upper[i]
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / (bb_mid[i] + 1e-10) < 0.05
        
        # === VOLUME CONFIRMATION ===
        volume_above_avg = volume[i] > 0.8 * (vol_sma[i] if not np.isnan(vol_sma[i]) else volume[i])
        volume_bullish = taker_ratio[i] > 0.55
        volume_bearish = taker_ratio[i] < 0.45
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === REGIME 1: RANGE (CHOP > 55) — MEAN REVERSION ===
        if regime_range and in_session:
            # Long: CRSI oversold + BB lower + HTF not bearish + volume
            if crsi_oversold and bb_break_lower and not htf_bearish:
                signal_strength = 2
                if crsi_extreme_oversold:
                    signal_strength += 1
                if volume_bullish and volume_above_avg:
                    signal_strength += 1
                if price_above_hma_4h:  # 4h still bullish
                    signal_strength += 1
                if signal_strength >= 3:
                    desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 5) / 5)
            
            # Short: CRSI overbought + BB upper + HTF not bullish + volume
            if crsi_overbought and bb_break_upper and not htf_bullish:
                if desired_signal == 0:
                    signal_strength = 2
                    if crsi_extreme_overbought:
                        signal_strength += 1
                    if volume_bearish and volume_above_avg:
                        signal_strength += 1
                    if price_below_hma_4h:  # 4h still bearish
                        signal_strength += 1
                    if signal_strength >= 3:
                        desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 5) / 5)
        
        # === REGIME 2: TREND (CHOP < 45) — TREND FOLLOW ===
        elif regime_trend and in_session:
            # Long: Pullback to BB mid + HTF bullish + volume
            if htf_bullish and close[i] < bb_mid[i] and close[i] > bb_lower[i]:
                if crsi_oversold:
                    signal_strength = 2
                    if volume_bullish and volume_above_avg:
                        signal_strength += 1
                    if price_above_hma_4h:
                        signal_strength += 1
                    if signal_strength >= 3:
                        desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 4) / 4)
            
            # Short: Rally to BB mid + HTF bearish + volume
            if htf_bearish and close[i] > bb_mid[i] and close[i] < bb_upper[i]:
                if desired_signal == 0 and crsi_overbought:
                    signal_strength = 2
                    if volume_bearish and volume_above_avg:
                        signal_strength += 1
                    if price_below_hma_4h:
                        signal_strength += 1
                    if signal_strength >= 3:
                        desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 4) / 4)
        
        # === REGIME 3: TRANSITION (45-55) — CONSERVATIVE ===
        else:
            # Only take extreme CRSI signals with strong HTF confirmation
            if crsi_extreme_oversold and htf_bullish and in_session:
                if volume_bullish:
                    desired_signal = position_size * 0.6
            elif crsi_extreme_overbought and htf_bearish and in_session:
                if desired_signal == 0 and volume_bearish:
                    desired_signal = -position_size * 0.6
        
        # === CAP SIGNAL TO MAX 0.30 ===
        if desired_signal > 0.30:
            desired_signal = 0.30
        elif desired_signal < -0.30:
            desired_signal = -0.30
        
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_4h or price_above_hma_1d):
                desired_signal = position_size
            elif position_side < 0 and (price_below_hma_4h or price_below_hma_1d):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.25:
                    desired_signal = 0.30
                elif desired_signal >= 0.15:
                    desired_signal = 0.20
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.25:
                    desired_signal = -0.30
                elif desired_signal <= -0.15:
                    desired_signal = -0.20
                else:
                    desired_signal = -0.15
        
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