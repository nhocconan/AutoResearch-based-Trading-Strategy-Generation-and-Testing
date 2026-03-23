#!/usr/bin/env python3
"""
Experiment #157: 1d Primary + 1w HTF — Dual Regime Strategy with Vol Spike Filter

Hypothesis: Daily timeframe with weekly trend filter can capture major moves while
avoiding whipsaw. Key innovations:

1) 1w HMA(21) for macro trend bias — only trade WITH weekly direction
2) Choppiness Index(14) regime switch — CHOP>55=range(mean revert), CHOP<40=trend(breakout)
3) Vol Spike Filter — ATR(7)/ATR(30) > 1.8 signals exhaustion → fade the move
4) Connors RSI(3,2,100) for mean reversion entries in range regime
5) Donchian(20) breakout for trend regime entries
6) ATR(14) stoploss at 2.5x — mandatory risk management
7) Position size: 0.25 base, 0.30 with full confluence

Why this might work:
- 1d timeframe = fewer trades (20-50/year), less fee drag
- Weekly trend filter avoids counter-trend trades in strong moves
- Vol spike filter catches exhaustion points (proven in 2022 crash)
- Dual regime adapts to market conditions (range vs trend)

Target: 25-50 trades/year per symbol, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_vol_chop_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    chop = np.zeros(len(close))
    mask = price_range > 0
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank(100)
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50,
        raw=False
    )
    percent_rank = percent_rank.fillna(50).values
    
    rsi_close_arr = rsi_close.fillna(50).values
    rsi_streak_arr = rsi_streak.fillna(50).values
    
    crsi = (rsi_close_arr + rsi_streak_arr + percent_rank) / 3.0
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend direction
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Vol spike ratio
    vol_spike_ratio = np.zeros(n)
    mask = atr_30 > 0
    vol_spike_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF TREND BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === VOL SPIKE FILTER ===
        # Vol spike > 1.8 suggests exhaustion → favor mean reversion
        vol_spike = vol_spike_ratio[i] > 1.8
        
        # === REGIME DETECTION ===
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 40.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME: Mean Reversion with CRSI ---
        if is_range:
            # Long: CRSI < 15 (extreme oversold) + weekly trend not bearish
            if crsi[i] < 15.0:
                if not price_below_hma_1w or vol_spike:
                    new_signal = POSITION_SIZE_BASE
                    # Add size if weekly aligned OR vol spike exhaustion
                    if price_above_hma_1w or vol_spike:
                        new_signal = POSITION_SIZE_MAX
            
            # Short: CRSI > 85 (extreme overbought) + weekly trend not bullish
            if crsi[i] > 85.0:
                if not price_above_hma_1w or vol_spike:
                    new_signal = -POSITION_SIZE_BASE
                    # Add size if weekly aligned OR vol spike exhaustion
                    if price_below_hma_1w or vol_spike:
                        new_signal = -POSITION_SIZE_MAX
        
        # --- TREND REGIME: Donchian Breakout with Weekly Confirmation ---
        if is_trend:
            # Long breakout: price breaks Donchian upper + weekly bullish
            if close[i] > donchian_upper[i-1] and price_above_hma_1w and not vol_spike:
                new_signal = POSITION_SIZE_BASE
            
            # Short breakdown: price breaks Donchian lower + weekly bearish
            if close[i] < donchian_lower[i-1] and price_below_hma_1w and not vol_spike:
                new_signal = -POSITION_SIZE_BASE
        
        # --- VOL SPIKE MEAN REVERSION (overrides regime) ---
        # If vol spike detected, fade the extreme move regardless of regime
        if vol_spike:
            # Long if price near Donchian lower + CRSI oversold
            if close[i] < donchian_lower[i-1] * 1.02 and crsi[i] < 25.0:
                new_signal = POSITION_SIZE_BASE
            
            # Short if price near Donchian upper + CRSI overbought
            if close[i] > donchian_upper[i-1] * 0.98 and crsi[i] > 75.0:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought
                if crsi[i] < 80.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold
                if crsi[i] > 20.0:
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
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals