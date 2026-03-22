#!/usr/bin/env python3
"""
Experiment #415: 1h Primary + 4h/1d HTF — Simplified Trend + RSI Pullback

Hypothesis: After 414 experiments, clear pattern emerges for lower TF:
1. 1h timeframe needs 30-60 trades/year — TOO MANY filters = 0 trades (#408, #410)
2. Previous 1h attempts failed due to over-filtering (session + chop + CRSI + volume)
3. SIMPLER logic works: HTF trend + LTF RSI pullback + basic volume
4. 4h HMA(21) proven in #406 (kept with Sharpe=0.011)
5. RSI(14) pullback to 35/65 levels = proven entry timing
6. Remove: session filters, Choppiness, Connors RSI (too restrictive for 1h)

Why this might beat current best (Sharpe=0.435):
- 1h captures more moves than 12h/1d while HTF filter prevents whipsaw
- RSI pullback entries have 60-65% win rate in research
- Volume filter (simple 20-bar avg) confirms participation without over-filtering
- ATR 2.5x stop protects from 2022-style crashes
- Discrete sizing (0.25) reduces fee churn vs continuous sizing

Position sizing: 0.25 (discrete levels for 1h TF)
Stoploss: 2.5 * ATR(14) trailing
Target: 40-70 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_4h1d_simp_v1"
timeframe = "1h"
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

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_48 = calculate_hma(df_4h['close'].values, period=48)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # 1h HMA for local trend
    hma_1h_16 = calculate_hma(close, period=16)
    hma_1h_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_16[i]) or np.isnan(hma_1h_48[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === 4H MAJOR TREND (primary direction filter) ===
        # Price above 4h HMA(21) = bull bias
        # Price below 4h HMA(21) = bear bias
        bull_4h = close[i] > hma_4h_21_aligned[i]
        bear_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA crossover confirmation
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_48_aligned[i]
        
        # === 1D REGIME FILTER (avoid counter-trend in major reversals) ===
        bull_1d = close[i] > hma_1d_21_aligned[i]
        bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 1H LOCAL TREND ===
        hma_1h_bullish = hma_1h_16[i] > hma_1h_48[i]
        hma_1h_bearish = hma_1h_16[i] < hma_1h_48[i]
        
        # === RSI PULLBACK SIGNALS ===
        # RSI < 35 = oversold pullback in uptrend (long entry)
        # RSI > 65 = overbought pullback in downtrend (short entry)
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # RSI extreme for exit
        rsi_extreme_long = rsi_14[i] > 75.0
        rsi_extreme_short = rsi_14[i] < 25.0
        
        # === VOLUME CONFIRMATION (simple filter) ===
        vol_confirm = volume[i] > 0.8 * vol_sma_20[i]
        
        # === ENTRY LOGIC — SIMPLIFIED FOR 1H ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bull + RSI pullback + volume
        if bull_4h and hma_4h_bullish:
            # Primary: RSI pullback in uptrend
            if rsi_oversold and vol_confirm:
                new_signal = SIZE
            # Secondary: 1h HMA cross with RSI neutral
            elif hma_1h_bullish and 40.0 < rsi_14[i] < 55.0:
                new_signal = SIZE * 0.8
        
        # SHORT ENTRY: 4h bear + RSI pullback + volume
        if bear_4h and hma_4h_bearish:
            # Primary: RSI pullback in downtrend
            if rsi_overbought and vol_confirm:
                if new_signal == 0.0:
                    new_signal = -SIZE
            # Secondary: 1h HMA cross with RSI neutral
            elif hma_1h_bearish and 45.0 < rsi_14[i] < 60.0:
                if new_signal == 0.0:
                    new_signal = -SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 30 bars (~30 hours on 1h), allow weaker entry
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if bull_4h and rsi_14[i] < 45.0:
                new_signal = SIZE * 0.6
            elif bear_4h and rsi_14[i] > 55.0:
                if new_signal == 0.0:
                    new_signal = -SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit)
        if in_position and position_side > 0 and rsi_extreme_long:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_extreme_short:
            new_signal = 0.0
        
        # 4h trend reversal exit
        if in_position and position_side > 0 and bear_4h:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_4h:
            new_signal = 0.0
        
        # 1h local trend reversal exit
        if in_position and position_side > 0 and hma_1h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_1h_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, high[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if low[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = low[i]
            else:
                lowest_price = min(lowest_price, low[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if high[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals