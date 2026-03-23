#!/usr/bin/env python3
"""
Experiment #109: 4h Primary + 1d HTF — KAMA Trend + Connors RSI Entries

Hypothesis: Previous experiments failed due to over-filtering (volume, complex regime detection).
This simplifies to proven components:

1) 1d HMA(21) for macro trend bias — only trade in trend direction
2) 4h KAMA(14) for adaptive trend following — KAMA adapts to volatility better than EMA/HMA
3) Connors RSI(3,2,100) for entry timing — CRSI<10 for long, CRSI>90 for short (mean reversion in trend)
4) ATR(14) trailing stop at 2.5x — locks profits, limits drawdown
5) Simple exit: CRSI crosses mid (50) or trend reversal

Why this should work:
- KAMA adapts to ranging vs trending markets automatically (no regime filter needed)
- CRSI has 75% win rate for mean reversion entries in trending markets
- 4h naturally produces 30-50 trades/year (optimal fee/trade balance)
- Simpler logic = more trades = better statistical significance
- 1d HMA filter prevents counter-trend trades in bear markets (2022 crash)

Position size: 0.25 base, 0.30 max with strong confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_crsi_hma_1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = close_s.diff(er_period).abs()
    noise = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / (noise + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.maximum(streak, 0)
    streak_loss = np.maximum(-streak, 0)
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank(100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100,
        raw=False
    )
    
    # CRSI
    crsi = (rsi + streak_rsi + percent_rank) / 3.0
    
    return crsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope (trend strength)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_14 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Simple SMA for additional trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_14[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_slope_positive = hma_1d_slope[i] > 0.3
        hma_slope_negative = hma_1d_slope[i] < -0.3
        hma_slope_flat = abs(hma_1d_slope[i]) <= 0.3
        
        # === 4h TREND FILTER (KAMA) ===
        kama_bullish = kama_14[i] > kama_50[i]
        kama_bearish = kama_14[i] < kama_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Long entry zone
        crsi_overbought = crsi[i] > 85  # Short entry zone
        crsi_neutral = 40 < crsi[i] < 60  # Exit zone
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 1d trend up/flat + 4h KAMA bullish + CRSI oversold + price above SMA200
        if price_above_hma_1d or hma_slope_flat:
            if kama_bullish and crsi_oversold and price_above_sma200:
                new_signal = POSITION_SIZE_BASE
                # Increase size if 1d slope strongly positive
                if hma_slope_positive:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Require: 1d trend down/flat + 4h KAMA bearish + CRSI overbought + price below SMA200
        if price_below_hma_1d or hma_slope_flat:
            if kama_bearish and crsi_overbought and price_below_sma200:
                new_signal = -POSITION_SIZE_BASE
                # Increase size if 1d slope strongly negative
                if hma_slope_negative:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if CRSI not yet at exit zone and trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI < 60 and 1d trend intact
                if crsi[i] < 60 and price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 and signals[i-1] != 0 else POSITION_SIZE_BASE
            elif position_side < 0:
                # Hold short if CRSI > 40 and 1d trend intact
                if crsi[i] > 40 and price_below_hma_1d:
                    new_signal = signals[i-1] if i > 0 and signals[i-1] != 0 else -POSITION_SIZE_BASE
        
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses down
            if price_below_hma_1d and hma_slope_negative:
                new_signal = 0.0
            # Exit long if KAMA turns bearish
            if kama_bearish:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses up
            if price_above_hma_1d and hma_slope_positive:
                new_signal = 0.0
            # Exit short if KAMA turns bullish
            if kama_bullish:
                new_signal = 0.0
        
        # === EXIT ON CRSI NEUTRAL (take profit) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 65:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 35:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals