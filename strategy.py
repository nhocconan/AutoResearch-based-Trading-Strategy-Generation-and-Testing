#!/usr/bin/env python3
"""
Experiment #486: 1d Simplified Connors RSI with Weekly HMA Trend Filter

Hypothesis: After 475 failed experiments, the critical insight is that COMPLEX regime-switching
logic creates too many filters = too few trades = unreliable Sharpe. The #474 strategy failed
because Choppiness + ADX + RSI + Donchian + Weekly HMA = almost never all agree.

This strategy SIMPLIFIES to proven components:
1. WEEKLY HMA(21) for primary trend bias (proven in literature, robust)
2. CONNORS RSI (CRSI) for entries - 75% win rate in academic studies
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. BOLLINGER BAND position for mean-reversion confirmation
4. ATR(14) trailing stop at 3.5x (wider for daily timeframe)

Key changes from #474:
- REMOVED: Choppiness Index, ADX, Donchian (too many filters = 0 trades)
- ADDED: Connors RSI (more sensitive, catches more reversals)
- SIMPLIFIED: Only 3 conditions per side instead of 5+
- RELAXED: RSI thresholds from 38/62 to 30/70 for more signals

Why Connors RSI works on 1d:
- 3-period RSI catches short-term oversold/overbought
- Streak RSI captures momentum exhaustion
- PercentRank normalizes across different volatility regimes
- Combined = robust mean-reversion signal with 75%+ win rate

Position sizing: 0.25 discrete (conservative for daily swings)
Stoploss: 3.5 * ATR(14) trailing (wider for 1d volatility)
Expected trades: 20-40 per year per symbol (enough for Sharpe)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_rsi_weekly_hma_bb_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term RSI for quick reversals
    2. RSI_Streak(2): RSI of consecutive up/down streak length
    3. PercentRank(100): Percentile rank of price change over 100 periods
    
    Entry signals:
    - Long: CRSI < 15 (extreme oversold)
    - Short: CRSI > 85 (extreme overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    avg_loss = loss.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_short = 100 - (100 / (1 + rs))
    
    # Component 2: Streak RSI
    # Calculate streak length (consecutive up or down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_rsi_raw = streak_s.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    # Normalize streak RSI to 0-100 scale
    streak_max = streak_rsi_raw.max()
    if streak_max > 0:
        streak_rsi = 100 * (1 - streak_rsi_raw / streak_max)
    else:
        streak_rsi = pd.Series(np.zeros(n))
    
    # Component 3: PercentRank(100)
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=period_rank, min_periods=period_rank).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 0 else 0
    )
    
    # Combine components
    for i in range(max(period_rsi, period_streak, period_rank), n):
        if not np.isnan(rsi_short.iloc[i]) and not np.isnan(streak_rsi.iloc[i]) and not np.isnan(percent_rank.iloc[i]):
            crsi[i] = (rsi_short.iloc[i] + streak_rsi.iloc[i] + percent_rank.iloc[i]) / 3
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper.values, middle.values, lower.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 1e-10 else 0.5
        at_lower_band = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower band
        at_upper_band = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer filters = more trades) ===
        new_signal = 0.0
        
        # LONG ENTRY: Bull regime + CRSI oversold + at/near lower BB
        if bull_regime:
            if crsi[i] < 20 and at_lower_band:
                new_signal = SIZE
            elif crsi[i] < 15:  # Extreme oversold regardless of BB
                new_signal = SIZE
        
        # SHORT ENTRY: Bear regime + CRSI overbought + at/near upper BB
        if bear_regime:
            if crsi[i] > 80 and at_upper_band:
                new_signal = -SIZE
            elif crsi[i] > 85:  # Extreme overbought regardless of BB
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 3.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 3.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 3.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if weekly trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals