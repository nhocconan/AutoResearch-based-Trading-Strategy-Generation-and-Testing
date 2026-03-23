#!/usr/bin/env python3
"""
Experiment #018: 30m Primary + 4h HTF — Connors RSI Mean Reversion with Volume

Hypothesis: 30m timeframe with LOOSE CRSI entries + 4h trend bias will generate 40-80 trades/year
while maintaining positive Sharpe through mean reversion edge.

CRITICAL LESSON from #008, #010: 30m strategies got Sharpe=0.000 (ZERO TRADES).
Entry conditions were TOO STRICT. This strategy GUARANTEES trades:
- CRSI < 25 or > 75 (not < 10 or > 90)
- Only 1 additional confluence filter required (not 2-3)
- No session filter (trades 24/7)
- Position size 0.25 (conservative for lower TF)

Components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. 4h HMA(21): Trend bias filter (loaded ONCE before loop)
3. Volume filter: volume > 0.7x 20-bar avg (loose)
4. ATR(14) stoploss: 2.5*ATR trailing

Why this works for 30m:
- HTF (4h) provides trend direction
- 30m only for entry timing within HTF trend
- LOOSE CRSI ensures trades are generated
- Small position size (0.25) controls drawdown
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_volume_4h_trend_v1"
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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = calculate_rsi(streak, period=streak_period)
    
    # Component 3: Percent Rank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i]
        current = close[i]
        count_below = np.sum(lookback < current)
        percent_rank[i] = count_below / rank_period * 100
    
    # CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 30m)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Need 100 for CRSI rank + 50 for other indicators
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(vol_avg[i]) or atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CRSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 25.0  # LOOSE (was < 10 in failed strategies)
        crsi_overbought = crsi[i] > 75.0  # LOOSE (was > 90 in failed strategies)
        
        # === VOLUME FILTER (LOOSE) ===
        volume_ok = volume[i] > vol_avg[i] * 0.7  # 70% of average is OK
        
        # === ENTRY LOGIC (LOOSE - only need 1 additional confluence) ===
        new_signal = 0.0
        
        # Long: CRSI oversold + (4h bullish OR volume confirms)
        if crsi_oversold:
            confluence_count = 0
            if price_above_hma_4h:
                confluence_count += 1
            if volume_ok:
                confluence_count += 1
            
            if confluence_count >= 1:  # Only need 1 additional filter!
                new_signal = POSITION_SIZE
        
        # Short: CRSI overbought + (4h bearish OR volume confirms)
        elif crsi_overbought:
            confluence_count = 0
            if price_below_hma_4h:
                confluence_count += 1
            if volume_ok:
                confluence_count += 1
            
            if confluence_count >= 1:  # Only need 1 additional filter!
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
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