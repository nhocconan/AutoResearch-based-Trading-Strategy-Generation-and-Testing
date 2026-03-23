#!/usr/bin/env python3
"""
Experiment #147: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Complex regime switching failed in #136, #139, #142, #146 (negative Sharpe).
Simple trend-following with pullback entries worked in #143 (Sharpe=0.115, +55% return).
This strategy uses PROVEN simple logic:

1) 1w HMA(21) for macro trend bias — only trade in direction of weekly trend
2) 1d HMA(16) vs HMA(48) crossover for entry timing
3) RSI(14) filter — enter on pullback (RSI 40-60 in uptrend, 40-60 in downtrend)
4) ATR(14) trailing stop at 2.5x — protects capital during reversals
5) Exit on HMA crossover reversal OR stoploss

Why this should work:
- 1d naturally produces 20-40 trades/year (low fee drag, matches Rule 10)
- Weekly trend filter avoids counter-trend trades (major failure mode)
- RSI pullback entries catch dips in trends (proven in mtf_hma_rsi_zscore_v1)
- Simpler than failed regime strategies — fewer conflicting filters = more trades
- HMA is faster than EMA, catches trends earlier

Position size: 0.25 base, 0.30 with strong confluence
Stoploss: 2.5*ATR trailing
Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50).values

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
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA CROSSOVER ===
        hma_cross_long = hma_16[i] > hma_48[i] and hma_16[i-1] <= hma_48[i-1]
        hma_cross_short = hma_16[i] < hma_48[i] and hma_16[i-1] >= hma_48[i-1]
        
        # === HMA TREND STATE (not just crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK FILTER ===
        # In uptrend: RSI 35-55 = pullback entry opportunity
        # In downtrend: RSI 45-65 = pullback entry opportunity
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry: weekly uptrend + HMA bullish + RSI pullback
        if price_above_hma_1w and hma_bullish and rsi_pullback_long:
            # Entry on HMA cross OR continuation
            if hma_cross_long or (hma_bullish and signals[i-1] == 0):
                new_signal = POSITION_SIZE_BASE
                # Stronger signal if RSI closer to 40 (deeper pullback)
                if rsi_14[i] < 45.0:
                    new_signal = POSITION_SIZE_MAX
        
        # Short entry: weekly downtrend + HMA bearish + RSI pullback
        if price_below_hma_1w and hma_bearish and rsi_pullback_short:
            # Entry on HMA cross OR continuation
            if hma_cross_short or (hma_bearish and signals[i-1] == 0):
                new_signal = -POSITION_SIZE_BASE
                # Stronger signal if RSI closer to 60 (shallower pullback in downtrend)
                if rsi_14[i] > 55.0:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if HMA still bullish
                if hma_bullish and price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if HMA still bearish
                if hma_bearish and price_below_hma_1w:
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit long if HMA crosses bearish OR price below weekly HMA
            if hma_cross_short or price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HMA crosses bullish OR price above weekly HMA
            if hma_cross_long or price_above_hma_1w:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            # Long take profit at RSI overbought
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            # Short take profit at RSI oversold
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
                # Position flip
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