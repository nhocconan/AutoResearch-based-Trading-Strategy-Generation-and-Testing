#!/usr/bin/env python3
"""
Experiment #007: 1d Primary + 1w HTF — Connors RSI Mean Reversion + HTF Trend

Hypothesis: After 6 failed experiments with complex multi-indicator strategies,
the pattern is clear: overfiltering kills performance. All 6 strategies had
negative Sharpe despite positive returns in some cases.

This strategy SIMPLIFIES drastically:
1. Connors RSI (CRSI) for mean-reversion entries — proven 75% win rate in literature
2. 1w HMA(21) for long-term trend bias — only trade WITH the weekly trend
3. ATR(14) stoploss — 2.5x ATR trailing stop
4. 1d timeframe — proven to work better than lower TFs (exp #003, #006 had +21-27% returns)

Why this might work where others failed:
- CRSI is specifically designed for mean-reversion in bear/range markets (2022, 2025)
- Single HTF (1w) avoids overfiltering from multiple timeframe conflicts
- Simpler entry logic = more trades, less chance of 0-trade failure
- 1d timeframe = 20-40 trades/year target, well within fee drag limits

Connors RSI Formula:
CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(3): 3-period RSI on close
- RSI_Streak(2): RSI on streak length (consecutive up/down days)
- PercentRank(100): percentile rank of today's return vs last 100 days

Entry Rules:
- Long: CRSI < 15 + price > 1w HMA(21) + 1w HMA sloping up
- Short: CRSI > 85 + price < 1w HMA(21) + 1w HMA sloping down
- Exit: CRSI crosses 50 OR stoploss hit

Position sizing: 0.30 discrete (conservative for 1d)
Target: 25-45 trades/year on 1d
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma1w_meanrev_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Standard RSI on 3-period close
    2. RSI_Streak(2): RSI applied to streak length (consecutive up/down days)
    3. PercentRank(100): Percentile rank of today's return vs last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI on streak length
    # Calculate streak: consecutive up (+1) or down (-1) days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    
    for i in range(1, n):
        if returns.iloc[i] > 0:
            if i > 0 and returns.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif returns.iloc[i] < 0:
            if i > 0 and returns.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Take absolute value for RSI calculation (streak magnitude)
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    
    # RSI on streak (using same RSI formula but on streak values)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = (-streak_delta).where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # Component 3: PercentRank of returns over last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period:i+1].dropna()
        if len(window_returns) > 0:
            current_return = returns.iloc[i]
            rank = (window_returns <= current_return).sum() / len(window_returns)
            percent_rank[i] = rank * 100.0
    
    # Combine all three components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for long-term trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # SMA200 for additional trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Need 200 for SMA + 100 for CRSI rank + buffer
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(sma_200[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-5] if i >= 5 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-5] if i >= 5 else False
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        price_above_sma_200 = close[i] > sma_200[i]
        price_below_sma_200 = close[i] < sma_200[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        crsi_neutral = 45.0 < crsi[i] < 55.0  # Exit zone
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # CRSI extreme oversold + price above weekly HMA + weekly HMA sloping up
        # OR: CRSI oversold + price above SMA200 (secondary confirmation)
        if crsi_oversold:
            if price_above_hma_1w and hma_1w_slope_bull:
                new_signal = POSITION_SIZE
            elif price_above_sma_200 and not hma_1w_slope_bear:
                # Weaker signal: only if not in strong weekly downtrend
                new_signal = POSITION_SIZE * 0.5  # Half size for weaker setup
        
        # --- SHORT ENTRY ---
        # CRSI extreme overbought + price below weekly HMA + weekly HMA sloping down
        # OR: CRSI overbought + price below SMA200 (secondary confirmation)
        if crsi_overbought:
            if price_below_hma_1w and hma_1w_slope_bear:
                new_signal = -POSITION_SIZE
            elif price_below_sma_200 and not hma_1w_slope_bull:
                # Weaker signal: only if not in strong weekly uptrend
                new_signal = -POSITION_SIZE * 0.5  # Half size for weaker setup
        
        # === EXIT ON CRSI MEAN REVERSION ===
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi[i] > 55.0:
                new_signal = 0.0  # Long exit: CRSI reverted to mean
            elif position_side < 0 and crsi[i] < 45.0:
                new_signal = 0.0  # Short exit: CRSI reverted to mean
            else:
                new_signal = signals[i-1] if i > 0 else 0.0  # Hold
        
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
        
        # === EXIT ON WEEKLY TREND FLIP ===
        if in_position and position_side > 0:
            if price_below_hma_1w and hma_1w_slope_bear:
                new_signal = 0.0  # Weekly trend turned bearish
        
        if in_position and position_side < 0:
            if price_above_hma_1w and hma_1w_slope_bull:
                new_signal = 0.0  # Weekly trend turned bullish
        
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