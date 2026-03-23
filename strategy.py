#!/usr/bin/env python3
"""
Experiment #062: 12h Primary + 1d/1w HTF — Dual Regime (Trend/Mean-Revert)

Hypothesis: 12h timeframe with Choppiness Index regime detection will adapt 
to market conditions better than pure trend or pure mean-reversion strategies.
Using Connors RSI for mean-reversion entries (proven 75% win rate) and 
Donchian breakout for trend entries, with 1d HMA for intermediate bias.

Key design decisions:
1) 12h primary = proven higher TF (exp #047, #053 kept despite low Sharpe)
2) CHOP regime switch = adapts to bull/bear/range automatically
3) Connors RSI = better than standard RSI for mean reversion (3-period RSI + streak)
4) Donchian(20) breakout = clear trend entry signals (ensures trades)
5) 1d HMA(21) = intermediate trend bias (not too laggy like 1w)
6) 1w HMA(21) = macro filter (prevents counter-trend in major moves)
7) Position size 0.30 (discrete, within 0.20-0.35 range)
8) Stoploss 2.5*ATR trailing (mandatory per rules)

Why this should beat Sharpe=0.486:
- Simpler logic than #053 (fewer conflicting filters = more trades)
- Connors RSI has proven edge in mean-reversion (academic literature)
- Dual regime ensures trades in both trending AND ranging markets
- 12h TF = fewer trades than 4h (less fee drag) but more than 1d (more opportunities)

Target: 25-40 trades/year, Sharpe > 0.5, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_connors_donchian_regime_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's price change over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_gain = np.maximum(streak, 0)
    streak_loss = np.abs(np.minimum(streak, 0))
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_rsi = np.zeros(n)
    for i in range(n):
        if avg_streak_loss[i] == 0:
            streak_rsi[i] = 100.0
        else:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_streak_gain[i] / (avg_streak_loss[i] + 1e-10)))
    
    # Component 3: PercentRank of price change
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: 100.0 * (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1]) if len(x) > 1 else 50.0
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D INTERMEDIATE BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        price_above_sma_200 = close[i] > sma_200[i]
        price_below_sma_200 = close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 61.8  # Range market (high choppiness)
        is_trending = chop_value < 38.2  # Trend market (low choppiness)
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CONNORS RSI MEAN REVERSION SIGNALS ===
        crsi_oversold = crsi[i] < 10.0  # Extreme oversold
        crsi_overbought = crsi[i] > 90.0  # Extreme overbought
        
        # === STANDARD RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Donchian Breakout + HTF Bias ---
        if is_trending:
            # Long: Donchian breakout + 1d HMA bullish + 1w confirms or neutral
            if breakout_long and price_above_hma_1d:
                if price_above_hma_1w or (not price_below_hma_1w):
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + 1d HMA bearish + 1w confirms or neutral
            elif breakout_short and price_below_hma_1d:
                if price_below_hma_1w or (not price_above_hma_1w):
                    new_signal = -POSITION_SIZE
        
        # --- RANGING REGIME: Connors RSI Mean Reversion ---
        elif is_ranging:
            # Long: CRSI extreme oversold + price above SMA200 (bullish bias)
            if crsi_oversold and price_above_sma_200:
                new_signal = POSITION_SIZE
            # Also allow long on standard RSI oversold for more trades
            elif rsi_oversold and price_above_sma_200:
                new_signal = POSITION_SIZE
            
            # Short: CRSI extreme overbought + price below SMA200 (bearish bias)
            elif crsi_overbought and price_below_sma_200:
                new_signal = -POSITION_SIZE
            # Also allow short on standard RSI overbought for more trades
            elif rsi_overbought and price_below_sma_200:
                new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME: Breakout with HTF confirmation (ensures trades) ---
        else:
            # Long: Donchian breakout + 1w bias
            if breakout_long and price_above_hma_1w:
                new_signal = POSITION_SIZE
            # Short: Donchian breakdown + 1w bias
            elif breakout_short and price_below_hma_1w:
                new_signal = -POSITION_SIZE
            # Fallback: CRSI extremes for mean reversion
            elif crsi_oversold:
                new_signal = POSITION_SIZE
            elif crsi_overbought:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if RSI not at opposite extreme (avoid premature exits)
            if position_side > 0 and rsi_14[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0:
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
        
        # === EXIT ON HTF TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1d and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d and price_above_hma_1w:
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