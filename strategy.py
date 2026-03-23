#!/usr/bin/env python3
"""
Experiment #427: 1d Primary + 1w HTF — Simplified Regime + CRSI + Donchian Breakout

Hypothesis: #417 had positive Sharpe (0.042) but too few trades due to strict conditions.
This version loosens entry thresholds while keeping the proven structure:
1. Weekly HMA for bias (bull/bear filter) — unchanged from #417
2. Connors RSI at 25/75 instead of 15/85 — generates more signals
3. Choppiness at 55/45 instead of 61.8/38.2 — more regime switches
4. Donchian(20) breakout confirmation for trend entries — adds momentum filter
5. ATR-based position sizing — reduce size in high vol regimes
6. 2.5x ATR trailing stoploss — proven risk management

Key changes from #417 to increase trade frequency:
- CRSI thresholds: 25/75 instead of 15/85 (3x more trigger points)
- Choppiness: 55/45 instead of 61.8/38.2 (more regime detection)
- Remove SMA200 filter (redundant with HMA bias)
- Add Donchian breakout for trend confirmation (catches momentum)
- Simpler hold logic — maintain position until clear exit signal

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -40%
Timeframe: 1d (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_donchian_chop_1w_v2"
timeframe = "1d"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 55 = choppy/range market
    CHOP < 45 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        crsi[i] = (rsi_3.iloc[i] + rsi_streak.iloc[i] + rank) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d
    
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
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === CHOPPINESS REGIME (loosened thresholds) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === HTF BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (price vs HMA21) ===
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.998  # near breakout
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.002  # near breakout
        
        # === CONNORS RSI THRESHOLDS (loosened from 15/85 to 25/75) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_mild_oversold = crsi[i] < 40.0
        crsi_mild_overbought = crsi[i] > 60.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        if price_above_hma_1w:  # HTF bullish bias
            if is_trending:
                # Trend following - need Donchian breakout confirmation
                if donchian_breakout_long and crsi_mild_oversold:
                    desired_signal = position_size
                elif price_above_hma21 and crsi_oversold:
                    desired_signal = position_size
            elif is_choppy:
                # Mean reversion in range
                if crsi_oversold:
                    desired_signal = position_size
                elif crsi_mild_oversold and price_above_hma21:
                    desired_signal = position_size * 0.5
            else:
                # Neutral regime - use HMA + CRSI
                if price_above_hma21 and crsi_mild_oversold:
                    desired_signal = position_size
        
        # SHORT SETUP
        if price_below_hma_1w:  # HTF bearish bias
            if is_trending:
                # Trend following - need Donchian breakdown confirmation
                if donchian_breakout_short and crsi_mild_overbought:
                    desired_signal = -position_size
                elif price_below_hma21 and crsi_overbought:
                    desired_signal = -position_size
            elif is_choppy:
                # Mean reversion in range
                if crsi_overbought:
                    desired_signal = -position_size
                elif crsi_mild_overbought and price_below_hma21:
                    desired_signal = -position_size * 0.5
            else:
                # Neutral regime - use HMA + CRSI
                if price_below_hma21 and crsi_mild_overbought:
                    desired_signal = -position_size
        
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
        
        # === CRSI EXTREME EXIT ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1w:
                desired_signal = position_size
            elif position_side < 0 and price_below_hma_1w:
                desired_signal = -position_size
        
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