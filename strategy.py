#!/usr/bin/env python3
"""
Experiment #593: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime

Hypothesis: 1d timeframe with 1w HTF bias should work well for bear/range markets (2025 test).
- Connors RSI (CRSI) for mean reversion entries (75% win rate in literature)
- Choppiness Index for regime detection (filter out bad trend trades in chop)
- 1w HMA(21) for major trend direction (prevents counter-trend disasters)
- ATR(14) 2.5x trailing stop for risk management

Why this might beat Sharpe=0.520:
- CRSI is proven for mean reversion in range markets (2025 test period is bear/range)
- CHOP filter prevents entering trend strategies during chop
- 1w HTF provides major trend bias protection
- 1d timeframe naturally limits trades to 20-50/year (fee efficient)
- Relaxed CRSI thresholds (15/85) ensure sufficient trade frequency

Entry Logic:
- CRSI < 15 + price > 1w_HMA = long (oversold in uptrend)
- CRSI > 85 + price < 1w_HMA = short (overbought in downtrend)
- CHOP > 61.8 = only mean reversion (no trend breakouts)
- CHOP < 38.2 = allow trend breakouts (Donchian 20)
- ATR 2.5x trailing stop on all positions

Position sizing: 0.25 discrete (conservative, max 0.40)
Target: >=30 trades/symbol train, >=3 test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_hma_1w_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
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
    Calculate Connors RSI (CRSI) - mean reversion indicator from literature.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close - short-term momentum
    2. RSI(2) of streak - streak duration (consecutive up/down days)
    3. PercentRank(100) - where current close ranks vs last 100 days
    
    CRSI < 15 = extremely oversold (long signal)
    CRSI > 85 = extremely overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) of streak
    # Streak = consecutive up/down days (+1 for up, -1 for down, 0 for flat)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: PercentRank(100) - where current close ranks in last 100
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100, raw=False
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    # CRSI = average of 3 components
    crsi = (rsi_close + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION ===
        is_chop_regime = chop_14[i] > 61.8
        is_trend_regime = chop_14[i] < 38.2
        
        # === 1W MAJOR TREND BIAS ===
        bull_bias_1w = close[i] > hma_1w_21_aligned[i]
        bear_bias_1w = close[i] < hma_1w_21_aligned[i]
        hma_1w_slope_bull = hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        hma_1w_slope_bear = hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Mean reversion in chop regime (CRSI extremes)
        if is_chop_regime:
            # Long: CRSI < 15 (oversold) + 1w bull bias
            if crsi[i] < 15.0 and bull_bias_1w:
                new_signal = POSITION_SIZE
            
            # Short: CRSI > 85 (overbought) + 1w bear bias
            elif crsi[i] > 85.0 and bear_bias_1w:
                new_signal = -POSITION_SIZE
        
        # Trend following in trend regime (Donchian breakout)
        elif is_trend_regime:
            # Long: Price breaks Donchian upper + 1w bull bias
            if close[i] > donchian_upper[i-1] and bull_bias_1w:
                new_signal = POSITION_SIZE if hma_1w_slope_bull else POSITION_SIZE * 0.7
            
            # Short: Price breaks Donchian lower + 1w bear bias
            elif close[i] < donchian_lower[i-1] and bear_bias_1w:
                new_signal = -POSITION_SIZE if hma_1w_slope_bear else -POSITION_SIZE * 0.7
        
        # Neutral regime - allow both mean reversion and breakouts with wider thresholds
        else:
            # Mean reversion with wider CRSI thresholds
            if crsi[i] < 12.0 and bull_bias_1w:
                new_signal = POSITION_SIZE * 0.7
            elif crsi[i] > 88.0 and bear_bias_1w:
                new_signal = -POSITION_SIZE * 0.7
            # Breakout with confirmation
            elif close[i] > donchian_upper[i-1] and bull_bias_1w and hma_1w_slope_bull:
                new_signal = POSITION_SIZE * 0.5
            elif close[i] < donchian_lower[i-1] and bear_bias_1w and hma_1w_slope_bear:
                new_signal = -POSITION_SIZE * 0.5
        
        # Hold position if already in one (avoid churn)
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
        
        # === EXIT ON BIAS FLIP ===
        if in_position and position_side > 0:
            if bear_bias_1w and hma_1w_slope_bear:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if bull_bias_1w and hma_1w_slope_bull:
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
                # Flip position
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