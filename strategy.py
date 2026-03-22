#!/usr/bin/env python3
"""
Experiment #507: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 454 failed strategies with complex filters (volspike/fisher/connors/chop),
go back to basics. The current best (Sharpe=0.435) uses simple HMA+RSI on 1d.

Key changes from failed experiments:
1. FEWER entry filters = MORE trades (critical: need >=30/symbol on train)
2. 1w HMA for major trend bias (not 1d, not 4h)
3. Simple RSI pullback entries (not Fisher, not Connors)
4. ATR trailing stop only (no complex exit logic)
5. Asymmetric sizing: 0.30 with trend, 0.20 against

Why this should work:
- 1d timeframe = 20-40 trades/year (low fee drag)
- 1w trend filter = avoids major counter-trend trades
- RSI pullback = proven entry timing
- Simple = fewer bugs, more reliable trade generation
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_simp_v1"
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE_WITH_TREND = 0.30
    LONG_SIZE_COUNTER = 0.20
    SHORT_SIZE_WITH_TREND = 0.30
    SHORT_SIZE_COUNTER = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Bull: price above 1w HMA21 AND 1w HMA21 > 1w HMA50
        bull_regime = (close[i] > hma_1w_21_aligned[i]) and (hma_1w_21_aligned[i] > hma_1w_50_aligned[i])
        # Bear: price below 1w HMA21 AND 1w HMA21 < 1w HMA50
        bear_regime = (close[i] < hma_1w_21_aligned[i]) and (hma_1w_21_aligned[i] < hma_1w_50_aligned[i])
        # Neutral: mixed signals
        neutral_regime = not bull_regime and not bear_regime
        
        # === 1D TREND CONFIRMATION ===
        hma_1d_bull = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bear = hma_1d_21[i] < hma_1d_50[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long pullback: RSI 35-50 in uptrend (not oversold, just pullback)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        # Short pullback: RSI 50-65 in downtrend (not overbought, just bounce)
        rsi_pullback_short = 50.0 <= rsi_14[i] <= 65.0
        # Extreme reversals
        rsi_oversold = rsi_14[i] < 30.0
        rsi_overbought = rsi_14[i] > 70.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — SIMPLIFIED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (3 conditions for frequency)
        # Condition 1: Bull regime + RSI pullback + above SMA200 (trend pullback)
        if bull_regime and rsi_pullback_long and above_sma200:
            new_signal = LONG_SIZE_WITH_TREND
        # Condition 2: Bull regime + RSI oversold (deep pullback)
        elif bull_regime and rsi_oversold:
            new_signal = LONG_SIZE_WITH_TREND
        # Condition 3: Neutral regime + RSI oversold + above SMA200 (mean reversion)
        elif neutral_regime and rsi_oversold and above_sma200:
            new_signal = LONG_SIZE_COUNTER
        # Condition 4: 1d HMA bull + RSI pullback (local trend)
        elif hma_1d_bull and rsi_pullback_long and above_sma200:
            new_signal = LONG_SIZE_COUNTER * 0.8
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: Bear regime + RSI pullback + below SMA200 (trend bounce)
            if bear_regime and rsi_pullback_short and below_sma200:
                new_signal = -SHORT_SIZE_WITH_TREND
            # Condition 2: Bear regime + RSI overbought (deep bounce)
            elif bear_regime and rsi_overbought:
                new_signal = -SHORT_SIZE_WITH_TREND
            # Condition 3: Neutral regime + RSI overbought + below SMA200 (mean reversion)
            elif neutral_regime and rsi_overbought and below_sma200:
                new_signal = -SHORT_SIZE_COUNTER
            # Condition 4: 1d HMA bear + RSI pullback (local trend)
            elif hma_1d_bear and rsi_pullback_short and below_sma200:
                new_signal = -SHORT_SIZE_COUNTER * 0.8
        
        # === STOPLOSS CHECK (3 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (simple, let winners run) ===
        # Exit long on major regime flip to bear
        if in_position and position_side > 0:
            if bear_regime and hma_1d_bear:
                new_signal = 0.0
            # Exit on RSI overbought extreme
            if rsi_14[i] > 80.0:
                new_signal = 0.0
        
        # Exit short on major regime flip to bull
        if in_position and position_side < 0:
            if bull_regime and hma_1d_bull:
                new_signal = 0.0
            # Exit on RSI oversold extreme
            if rsi_14[i] < 20.0:
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