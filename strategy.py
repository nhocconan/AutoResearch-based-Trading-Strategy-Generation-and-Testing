#!/usr/bin/env python3
"""
Experiment #006: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime

Hypothesis: After 5 failed experiments, returning to proven patterns from research.
Connors RSI has documented 75% win rate on BTC/ETH mean reversion.
Choppiness Index filters regime to avoid trend-chopping losses.

Key differences from failed attempts:
1. LOOSE entry conditions (CRSI < 25 OR > 75, not extreme <10/>90)
2. Choppiness regime switch: mean revert in chop (CHOP>50), trend in clean (CHOP<40)
3. 1d HMA for directional bias only (not strict filter)
4. Position size 0.30 (proven range from #001 success)
5. Simple ATR stoploss (2.5x) — no complex trailing

Why this might work:
- CRSI worked in #001 (Sharpe=0.366, best so far)
- 12h TF naturally limits trades to 20-50/year (fee-efficient)
- Regime-adaptive: different logic for chop vs trend
- Entry conditions loose enough to guarantee trades

Entry conditions (LOOSE — must generate trades):
- Long: CRSI < 25 AND (CHOP > 50 OR 1d HMA bullish)
- Short: CRSI > 75 AND (CHOP > 50 OR 1d HMA bearish)
- EITHER regime condition works (not both required)

Stoploss: 2.5*ATR, signal→0 when hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d_v1"
timeframe = "12h"
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of current return vs last N returns
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of streaks
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - how current return ranks vs last 100
    returns = close_s.pct_change()
    percent_rank = np.zeros(n) * np.nan
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if not np.isnan(current):
            rank = (window < current).sum() / len(window)
            percent_rank[i] = rank * 100.0
    
    percent_rank_s = pd.Series(percent_rank)
    
    # CRSI = average of 3 components
    crsi = (pd.Series(rsi_short) + rsi_streak + percent_rank_s) / 3.0
    
    return crsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # ATR
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    
    # Sum of ATR over period
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100.0 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    return chop.values

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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 50.0  # Range market
        is_trending = chop[i] < 40.0  # Trend market
        
        # === CRSI EXTREMES (LOOSE thresholds for trades) ===
        crsi_oversold = crsi[i] < 25.0  # Long entry
        crsi_overbought = crsi[i] > 75.0  # Short entry
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # In choppy market: mean reversion (CRSI oversold)
        chop_long = is_choppy and crsi_oversold
        
        # In trending market: only long if 1d HMA bullish
        trend_long = is_trending and crsi_oversold and (hma_1d_slope_bull or price_above_hma_1d)
        
        # Either regime condition works
        if chop_long or trend_long:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # In choppy market: mean reversion (CRSI overbought)
        chop_short = is_choppy and crsi_overbought
        
        # In trending market: only short if 1d HMA bearish
        trend_short = is_trending and crsi_overbought and (hma_1d_slope_bear or price_below_hma_1d)
        
        # Either regime condition works
        if chop_short or trend_short:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR) ===
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
        
        # === EXIT ON REGIME/TREND FLIP ===
        # Exit long if trend turns strongly bearish
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d and is_trending:
                new_signal = 0.0
        
        # Exit short if trend turns strongly bullish
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d and is_trending:
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