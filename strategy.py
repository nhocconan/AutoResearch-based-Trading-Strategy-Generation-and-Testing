#!/usr/bin/env python3
"""
Experiment #429: 4h Primary + 1d HTF — Vol Spike Mean Reversion + HMA Trend

Hypothesis: After 388 failed experiments, clear patterns emerge:
1. Strategies with Sharpe=0.000 (#418, #420, #425, #428) had ZERO trades - too many filters
2. Vol spike mean reversion has research backing (ATR ratio > 1.8 = panic capitulation)
3. 4h TF balances trade frequency (30-60/year) with signal quality
4. 1d HMA(21) for trend bias prevents counter-trend trades in crash
5. Simpler entry conditions = guaranteed trades (address #1 failure mode)

Why this might beat Sharpe=0.435:
- Vol spike captures panic bottoms (2022 crash, 2025 dips) with high win rate
- BB(20, 2.5) extreme = statistically significant oversold/overbought
- HMA trend filter prevents catching falling knives
- Frequency boost ensures >=30 trades/symbol on train
- Fewer conflicting filters = more consistent signal generation

Position sizing: 0.25-0.30 discrete levels (max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 4h, >=30 trades/symbol train, >=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_bb_hma_1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_bollinger(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with wider std for extreme detection."""
    close_s = pd.Series(close)
    mid = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid.values, upper.values, lower.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100)
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    for i in range(rank_period, n):
        window = pct_change.iloc[i-rank_period:i]
        current = pct_change.iloc[i]
        if not np.isnan(current) and len(window) > 0:
            rank = (window < current).sum() / len(window)
            percent_rank.iloc[i] = rank * 100.0
    
    crsi = (rsi_3.values + rsi_streak.values + percent_rank.values) / 3.0
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.5)
    
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close)
    
    # Volatility spike ratio (panic indicator)
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(atr_ratio[i]):
            continue
        
        # === 1D TREND BIAS (major direction) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY SPIKE (panic/capitulation) ===
        # Lowered from 2.0 to 1.8 for more signals
        vol_spike = atr_ratio[i] > 1.8
        vol_normal = atr_ratio[i] < 1.3
        
        # === BOLLINGER EXTREMES (wide bands for extremes) ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        bb_mid_cross_up = close[i] > bb_mid[i]
        bb_mid_cross_down = close[i] < bb_mid[i]
        
        # === RSI EXTREMES (relaxed for more trades) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === HMA TREND (local 4h) ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        new_signal = 0.0
        bars_since_trade = i - last_trade_bar
        
        # === LONG ENTRY (simplified - fewer filters for more trades) ===
        if bull_regime:
            # Vol spike + BB oversold = panic buy (primary signal)
            if vol_spike and bb_oversold:
                new_signal = LONG_SIZE
            # RSI oversold + HMA bullish (secondary)
            elif rsi_oversold and hma_bullish:
                new_signal = LONG_SIZE * 0.8
            # CRSI extreme oversold (tertiary)
            elif crsi_oversold:
                new_signal = LONG_SIZE * 0.7
            # Simple HMA bullish pullback
            elif hma_bullish and rsi_14[i] < 50.0:
                new_signal = LONG_SIZE * 0.6
        
        # === SHORT ENTRY (simplified) ===
        if bear_regime:
            # Vol spike + BB overbought = panic sell
            if vol_spike and bb_overbought:
                new_signal = -SHORT_SIZE
            # RSI overbought + HMA bearish
            elif rsi_overbought and hma_bearish:
                new_signal = -SHORT_SIZE * 0.8
            # CRSI extreme overbought
            elif crsi_overbought:
                new_signal = -SHORT_SIZE * 0.7
            # Simple HMA bearish rally
            elif hma_bearish and rsi_14[i] > 50.0:
                new_signal = -SHORT_SIZE * 0.6
        
        # === FREQUENCY BOOST (CRITICAL - prevent 0 trades) ===
        # If no trade for 15 bars (~2.5 days on 4h), enter on weaker signal
        if bars_since_trade > 15 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_14[i] < 45.0:
                new_signal = LONG_SIZE * 0.5
            elif bear_regime and rsi_14[i] > 55.0:
                new_signal = -SHORT_SIZE * 0.5
            elif hma_bullish and crsi[i] < 40.0:
                new_signal = LONG_SIZE * 0.5
            elif hma_bearish and crsi[i] > 60.0:
                new_signal = -SHORT_SIZE * 0.5
        
        # === EXIT CONDITIONS ===
        # Take profit on mean reversion (cross BB mid)
        if in_position and position_side > 0 and bb_mid_cross_up:
            new_signal = 0.0
        if in_position and position_side < 0 and bb_mid_cross_down:
            new_signal = 0.0
        
        # CRSI exhaustion (take profit)
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (4h HMA cross)
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals