#!/usr/bin/env python3
"""
Experiment #133: 1d Primary + 1w HTF — Connors RSI + KAMA Trend + Choppiness Regime

Hypothesis: Previous Donchian breakout strategies failed because breakouts whipsaw in 
ranging markets. This uses a dual-regime approach with Connors RSI for precise entries:

1) Connors RSI (CRSI) for mean-reversion entries - proven 75% win rate in research
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 15 (extreme oversold), Short: CRSI > 85 (extreme overbought)

2) KAMA (Kaufman Adaptive Moving Average) trend filter - adapts to volatility
   Only long if price > KAMA(21), only short if price < KAMA(21)
   KAMA responds slower in chop, faster in trends - perfect for regime detection

3) Choppiness Index regime switch:
   CHOP > 50 = ranging (use CRSI mean-reversion)
   CHOP < 50 = trending (use KAMA trend-follow with pullback entries)

4) 1w HMA(21) macro bias - only trade in direction of weekly trend
   Prevents counter-trend trades during major moves (like 2022 crash)

5) ATR(14) trailing stop at 2.5x - locks profits, limits drawdown

Why this should beat previous attempts:
- CRSI is more sensitive than regular RSI (uses 3-period instead of 14)
- KAMA adapts to market conditions automatically (no manual regime detection)
- Choppiness filter prevents trend strategies in chop and vice versa
- 1d timeframe = 25-40 trades/year naturally (low fee drag)
- Simpler than dual-regime strategies that failed (#121, #122)

Position size: 0.25 base, 0.30 with strong confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_kama_chop_regime_1w_v1"
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

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    noise = pd.Series(close).diff().abs()
    signal = pd.Series(close).diff(period).abs()
    er = signal / (noise.rolling(window=period, min_periods=period).sum() + 1e-10)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]
    
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_rsi_streak(close, period=2):
    """Calculate RSI of consecutive up/down streaks."""
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    delta = streak_s.diff()
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs))
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank - where current close ranks in last N periods."""
    pr = np.zeros(len(close))
    
    for i in range(period, len(close)):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / (period - 1) * 100.0
        pr[i] = rank
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI."""
    rsi_3 = calculate_rsi(close, period=rsi_period)
    rsi_streak = calculate_rsi_streak(close, period=streak_period)
    percent_rank = calculate_percent_rank(close, period=pr_period)
    
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(len(close))
    for i in range(period, len(close)):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(tr[i-period+1:i+1])
        
        if highest_high - lowest_low > 0:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21 = calculate_kama(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_21[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND FILTER (KAMA) ===
        price_above_kama = close[i] > kama_21[i]
        price_below_kama = close[i] < kama_21[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_ranging = chop[i] > 50.0
        is_trending = chop[i] <= 50.0
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_neutral = 15.0 <= crsi[i] <= 85.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Ranging regime: CRSI mean-reversion (oversold + price above KAMA)
        if is_ranging:
            if crsi_oversold and price_above_kama and price_above_hma_1w:
                new_signal = POSITION_SIZE_BASE
                if crsi[i] < 10.0:  # Extreme oversold
                    new_signal = POSITION_SIZE_MAX
        
        # Trending regime: Pullback to KAMA in uptrend
        if is_trending:
            if price_above_kama and price_above_hma_1w and crsi[i] < 40.0:
                # Pullback entry in uptrend
                new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # Ranging regime: CRSI mean-reversion (overbought + price below KAMA)
        if is_ranging:
            if crsi_overbought and price_below_kama and price_below_hma_1w:
                new_signal = -POSITION_SIZE_BASE
                if crsi[i] > 90.0:  # Extreme overbought
                    new_signal = -POSITION_SIZE_MAX
        
        # Trending regime: Pullback to KAMA in downtrend
        if is_trending:
            if price_below_kama and price_below_hma_1w and crsi[i] > 60.0:
                # Pullback entry in downtrend
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold long if CRSI not overbought and trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if not crsi_overbought and price_above_kama:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if not crsi_oversold and price_below_kama:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if highest_since_entry == 0.0:
                highest_since_entry = close[i]
            else:
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
            if price_below_hma_1w:
                new_signal = 0.0
            if crsi_overbought:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w:
                new_signal = 0.0
            if crsi_oversold:
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