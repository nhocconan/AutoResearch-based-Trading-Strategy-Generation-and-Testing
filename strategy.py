#!/usr/bin/env python3
"""
Experiment #117: 1d Primary + 1w HTF — Simplified Donchian + Connors RSI

Hypothesis: #107 had +7.8% return but -0.78 Sharpe due to volume filter being too
restrictive on daily data and poor exit timing. This version:

1) Remove volume filter (too restrictive on 1d - daily volume varies wildly)
2) Add Connors RSI for entry timing (proven on ETH with Sharpe +0.923 in research)
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 20 (oversold pullback in uptrend)
   - Short when CRSI > 80 (overbought pullback in downtrend)
3) Keep 1w HMA(21) for macro bias (proven effective)
4) Keep Donchian(20) breakout but enter on pullback, not breakout peak
5) ATR(14) 2.5x trailing stop (proven effective)
6) Asymmetric sizing: 0.30 with trend, 0.20 counter-trend

Why this should work:
- Connors RSI catches pullbacks in trends (better entry than breakout peak)
- No volume filter = more trades (20-40/year target)
- Simpler logic = more robust across BTC/ETH/SOL
- 1w HMA prevents counter-trend trades in bear markets

Position size: 0.25-0.30 discrete levels
Stoploss: 2.5*ATR trailing
Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_crsi_hma_1w_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-day high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: rank of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak_abs[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 - streak_abs[i] * 10)
    
    # Percent Rank (today's return vs last 100 days)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100
    
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current)
        percent_rank[i] = (rank / rank_period) * 100
    
    # Combine into CRSI
    for i in range(rank_period, n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1w HMA slope (trend strength)
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] != 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    POSITION_SIZE_TREND = 0.30
    POSITION_SIZE_COUNTER = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        hma_slope_positive = hma_1w_slope[i] > 0.3
        hma_slope_negative = hma_1w_slope[i] < -0.3
        
        # === 1d TREND FILTER ===
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 25  # Long entry zone
        crsi_overbought = crsi[i] > 75  # Short entry zone
        
        # === DONCHIAN POSITION ===
        above_donchian_mid = close[i] > donchian_mid[i]
        below_donchian_mid = close[i] < donchian_mid[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Primary: 1w trend up + 1d trend up + CRSI oversold (pullback entry)
        if price_above_hma_1w and hma_1d_bullish and crsi_oversold:
            new_signal = POSITION_SIZE_TREND
        # Secondary: 1w flat + 1d bullish + CRSI very oversold
        elif abs(hma_1w_slope[i]) <= 0.3 and hma_1d_bullish and crsi[i] < 15:
            new_signal = POSITION_SIZE_COUNTER
        # Breakout entry: price breaks Donchian high + trend aligned
        elif price_above_hma_1w and hma_1d_bullish and close[i] > donchian_upper[i-1]:
            if crsi[i] < 50:  # Not overbought
                new_signal = POSITION_SIZE_TREND
        
        # --- SHORT ENTRY ---
        # Primary: 1w trend down + 1d trend down + CRSI overbought (pullback entry)
        if price_below_hma_1w and hma_1d_bearish and crsi_overbought:
            new_signal = -POSITION_SIZE_TREND
        # Secondary: 1w flat + 1d bearish + CRSI very overbought
        elif abs(hma_1w_slope[i]) <= 0.3 and hma_1d_bearish and crsi[i] > 85:
            new_signal = -POSITION_SIZE_COUNTER
        # Breakout entry: price breaks Donchian low + trend aligned
        elif price_below_hma_1w and hma_1d_bearish and close[i] < donchian_lower[i-1]:
            if crsi[i] > 50:  # Not oversold
                new_signal = -POSITION_SIZE_TREND
        
        # === HOLD POSITION LOGIC ===
        # Hold long if still above Donchian mid and 1w trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if close[i] > donchian_mid[i] and price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if close[i] < donchian_mid[i] and price_below_hma_1w:
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_1w and hma_slope_negative:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if close[i] < donchian_lower[i-1]:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w and hma_slope_positive:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if close[i] > donchian_upper[i-1]:
                new_signal = 0.0
        
        # === TAKE PROFIT ON CRSI EXTREME ===
        if in_position and position_side > 0 and crsi[i] > 85:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 15:
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