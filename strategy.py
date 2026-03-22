#!/usr/bin/env python3
"""
Experiment #534: 1d Regime-Adaptive with Weekly HMA Bias + Connors RSI

Hypothesis: After 500+ failed experiments, the pattern is clear:
1. Pure trend-following fails in bear/range markets (2022 crash, 2025 bear)
2. Pure mean-reversion misses sustained moves
3. 1d timeframe needs regime detection to switch modes appropriately
4. Weekly HMA provides stable trend bias without whipsaw
5. Connors RSI (not standard RSI) has 75% win rate for mean-reversion entries
6. Choppiness Index distinguishes trending vs ranging regimes
7. Loose entry conditions ensure 10+ trades minimum

Why this should work on 1d:
- 1d has ~365 bars/year = fewer trades but higher quality
- Regime-adaptive: mean-revert in chop, trend-follow in clean trends
- Weekly HMA bias prevents counter-trend entries (major failure mode)
- Connors RSI < 10 / > 90 are rare but high-probability setups
- 2.5*ATR stoploss protects against 2022-style crashes
- Discrete sizing (0.30) limits drawdown during crashes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_weekly_hma_connors_rsi_chop_atr_v1"
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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long signal: CRSI < 10 (oversold)
    Short signal: CRSI > 90 (overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI component
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measures consecutive up/down days
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
            streak_rsi[i] = 100 * (streak_abs[i] / max(1, streak_abs[max(0, i-streak_period):i+1].max()))
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (1 - streak_abs[i] / max(1, streak_abs[max(0, i-streak_period):i+1].max()))
        else:
            streak_rsi[i] = 50
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - where current return ranks vs last 100 days
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns_window = np.diff(close[max(0, i-rank_period):i+1]) / close[max(0, i-rank_period):i]
        current_return = (close[i] - close[i-1]) / close[i-1] if i > 0 else 0
        pct_rank[i] = 100 * np.sum(returns_window <= current_return) / max(1, len(returns_window))
    
    # Combine components
    crsi = (rsi_fast + streak_rsi + pct_rank) / 3
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean-revert)
    CHOP < 38.2 = trending market (trend-follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True range sum over period
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high - Lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    hhll = hh - ll
    
    # CHOP formula
    chop = 100 * np.log10(tr_sum / hhll.replace(0, np.inf)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50)
    
    return chop.values

def calculate_sma(close, period=200):
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
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 55  # Range market (mean-revert mode)
        is_trending = chop[i] < 45  # Trending market (trend-follow mode)
        
        # === WEEKLY HMA TREND BIAS ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Loose threshold for more trades
        crsi_overbought = crsi[i] > 85  # Loose threshold for more trades
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # MODE 1: CHOPPY MARKET - Mean Reversion
        if is_choppy:
            # Long: CRSI oversold + above weekly HMA (bullish bias) + above SMA200
            if crsi_oversold and (bull_bias or above_sma200):
                new_signal = SIZE
            
            # Short: CRSI overbought + below weekly HMA (bearish bias) + below SMA200
            elif crsi_overbought and (bear_bias or below_sma200):
                new_signal = -SIZE
        
        # MODE 2: TRENDING MARKET - Trend Following
        elif is_trending:
            # Long: Bullish weekly bias + CRSI not overbought (pullback entry)
            if bull_bias and crsi[i] < 70 and above_sma200:
                new_signal = SIZE
            
            # Short: Bearish weekly bias + CRSI not oversold (pullback entry)
            elif bear_bias and crsi[i] > 30 and below_sma200:
                new_signal = -SIZE
        
        # MODE 3: NEUTRAL - Wait for clear signal
        else:
            # Only take extreme CRSI signals with strong weekly bias
            if crsi_oversold and bull_bias:
                new_signal = SIZE
            elif crsi_overbought and bear_bias:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === WEEKLY HMA REVERSAL EXIT ===
        # Exit if weekly HMA flips against position (major trend change)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
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