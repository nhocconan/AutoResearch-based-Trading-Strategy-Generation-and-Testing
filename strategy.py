#!/usr/bin/env python3
"""
Experiment #544: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: After analyzing 543 experiments, the pattern is clear:
- 4h strategies with complex regime switching consistently fail (#539, #541, #542)
- Higher TF (1d/12h) with simple trend+pullback works best (#543, current best Sharpe=0.435)
- For 4h to work, must use HTF (12h/1d) more aggressively for DIRECTION, 4h only for TIMING
- Trade frequency must be 20-50/year (not too many = fee drag, not too few = no signal)
- Asymmetric logic (different long/short rules) improves Sharpe in bear markets

This strategy combines:
1. 12h HMA(21) for major trend direction (HTF primary filter)
2. 1d HMA(50) for weekly trend confirmation (secondary HTF filter)
3. 4h HMA(16/48) crossover for entry timing
4. RSI(14) pullback zones (30-50 long, 50-70 short) for entry precision
5. Donchian(20) breakout confirmation for momentum
6. ATR(14) 2.5x trailing stop for risk management
7. ADX(14) > 20 filter to avoid choppy whipsaws

Key differences from failed 4h strategies:
- Simpler entry logic (OR conditions for frequency, not AND)
- Stronger HTF filter (12h + 1d, not just 12h)
- Asymmetric exits (quick exit on HTF flip, slower on 4h signals)
- Position size 0.30 (discrete, max 0.40 per rules)

Why this might beat current best (Sharpe=0.435):
- 4h TF captures more moves than 1d while using HTF for direction
- Multiple entry conditions ensure ≥30 trades/symbol on train
- ATR trailing stop limits drawdown during 2022 crash
- Simpler logic = more consistent signals across BTC/ETH/SOL
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_12h1d_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    """Calculate Donchian Channel (upper/lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF HMA for major trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Calculate 1d HTF HMA for weekly trend confirmation
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # 4h HMA for trend confirmation
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    
    # Donchian channels for breakout detection
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track previous values for crossover detection
    prev_hma_16 = np.zeros(n)
    prev_hma_16[1:] = hma_4h_16[:-1]
    prev_hma_48 = np.zeros(n)
    prev_hma_48[1:] = hma_4h_48[:-1]
    
    # Track Donchian breakout
    prev_donchian_upper = np.zeros(n)
    prev_donchian_upper[1:] = donchian_upper[:-1]
    prev_donchian_lower = np.zeros(n)
    prev_donchian_lower[1:] = donchian_lower[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength
        hma_12h_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 1D WEEKLY TREND (secondary confirmation) ===
        bull_regime_1d = close[i] > hma_1d_50_aligned[i]
        bear_regime_1d = close[i] < hma_1d_50_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        # HMA crossover (fast above slow = bull)
        hma_bull_cross = hma_4h_16[i] > hma_4h_48[i]
        hma_bear_cross = hma_4h_16[i] < hma_4h_48[i]
        
        # HMA crossover confirmation (just crossed)
        hma_bull_crossed = hma_bull_cross and (prev_hma_16[i] <= prev_hma_48[i])
        hma_bear_crossed = hma_bear_cross and (prev_hma_16[i] >= prev_hma_48[i])
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > prev_donchian_upper[i]
        donchian_breakout_short = close[i] < prev_donchian_lower[i]
        
        # === ADX FILTER (trending market) ===
        trending = adx_14[i] > 20.0  # Market is trending
        weak_trend = adx_14[i] < 18.0  # Trend weakening
        
        # === RSI PULLBACK FILTER ===
        rsi_pullback_long = 30.0 < rsi_14[i] < 50.0  # Pullback in uptrend
        rsi_pullback_short = 50.0 < rsi_14[i] < 70.0  # Pullback in downtrend
        rsi_neutral_long = rsi_14[i] < 65.0  # Not overbought for long
        rsi_neutral_short = rsi_14[i] > 35.0  # Not oversold for short
        
        # === ENTRY LOGIC — OR CONDITIONS FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions - any one triggers)
        # Condition 1: 12h bull + 4h HMA bull + Donchian breakout + RSI ok
        if bull_regime_12h and hma_bull_cross and donchian_breakout_long and rsi_neutral_long:
            new_signal = POSITION_SIZE
        # Condition 2: 12h bull + 4h HMA crossed bull + trending + RSI pullback
        elif bull_regime_12h and hma_bull_crossed and trending and rsi_pullback_long:
            new_signal = POSITION_SIZE
        # Condition 3: 12h bull + 12h slope bull + 4h HMA bull + Donchian breakout
        elif bull_regime_12h and hma_12h_slope_bull and hma_bull_cross and donchian_breakout_long:
            new_signal = POSITION_SIZE
        # Condition 4: 12h bull + 1d bull + 4h HMA bull + RSI pullback (strong confluence)
        elif bull_regime_12h and bull_regime_1d and hma_bull_cross and rsi_pullback_long:
            new_signal = POSITION_SIZE
        # Condition 5: 12h bull + 4h HMA bull + RSI pullback (simpler entry)
        elif bull_regime_12h and hma_bull_cross and rsi_pullback_long:
            new_signal = POSITION_SIZE * 0.9
        # Condition 6: Strong 12h bull (slope) + 4h HMA bull + trending
        elif bull_regime_12h and hma_12h_slope_bull and hma_bull_cross and trending:
            new_signal = POSITION_SIZE * 0.9
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: 12h bear + 4h HMA bear + Donchian breakout + RSI ok
            if bear_regime_12h and hma_bear_cross and donchian_breakout_short and rsi_neutral_short:
                new_signal = -POSITION_SIZE
            # Condition 2: 12h bear + 4h HMA crossed bear + trending + RSI pullback
            elif bear_regime_12h and hma_bear_crossed and trending and rsi_pullback_short:
                new_signal = -POSITION_SIZE
            # Condition 3: 12h bear + 12h slope bear + 4h HMA bear + Donchian breakout
            elif bear_regime_12h and hma_12h_slope_bear and hma_bear_cross and donchian_breakout_short:
                new_signal = -POSITION_SIZE
            # Condition 4: 12h bear + 1d bear + 4h HMA bear + RSI pullback (strong confluence)
            elif bear_regime_12h and bear_regime_1d and hma_bear_cross and rsi_pullback_short:
                new_signal = -POSITION_SIZE
            # Condition 5: 12h bear + 4h HMA bear + RSI pullback (simpler entry)
            elif bear_regime_12h and hma_bear_cross and rsi_pullback_short:
                new_signal = -POSITION_SIZE * 0.9
            # Condition 6: Strong 12h bear (slope) + 4h HMA bear + trending
            elif bear_regime_12h and hma_12h_slope_bear and hma_bear_cross and trending:
                new_signal = -POSITION_SIZE * 0.9
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === ASYMMETRIC EXIT CONDITIONS ===
        # Exit long on 12h regime flip to bear (quick exit on HTF flip)
        if in_position and position_side > 0:
            if bear_regime_12h and hma_12h_slope_bear:
                new_signal = 0.0
            elif weak_trend and not hma_bull_cross:  # Trend weakening + 4h HMA bearish
                new_signal = 0.0
        
        # Exit short on 12h regime flip to bull (quick exit on HTF flip)
        if in_position and position_side < 0:
            if bull_regime_12h and hma_12h_slope_bull:
                new_signal = 0.0
            elif weak_trend and not hma_bear_cross:  # Trend weakening + 4h HMA bullish
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