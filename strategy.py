#!/usr/bin/env python3
"""
Experiment #112: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend with HMA Filter

Hypothesis: Previous complex regime strategies failed due to overfitting and lag.
KAMA (Kaufman Adaptive Moving Average) adapts to market volatility automatically:
- Fast response in strong trends (high Efficiency Ratio)
- Slow response in choppy markets (low Efficiency Ratio)

This strategy uses:
1) 1w HMA(21) for macro trend bias — only trade in direction of weekly trend
2) 1d HMA(21) for intermediate trend confirmation
3) 12h KAMA(10,2,30) for adaptive entry signals — crossovers with price
4) ATR(14) trailing stop at 2.5x — locks profits, limits drawdown
5) Simple exit: KAMA cross back OR stoploss OR weekly trend reversal

Why this should work on 12h:
- KAMA reduces whipsaws in ranging markets (common in 2022-2023)
- 12h produces ~30-50 trades/year naturally (low fee drag)
- Dual HTF filter (1w + 1d) prevents counter-trend trades
- Simpler logic than regime-switching = more robust across BTC/ETH/SOL
- Proven KAMA success on 4h should translate well to 12h

Position size: 0.25 base, 0.30 with strong HTF confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_hma_trend_1d1w_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        net_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA slope (trend strength)
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] != 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate 1d HMA slope
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 12h indicators (pre-compute before loop for performance)
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_12h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_12h[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        hma_1w_strong_up = hma_1w_slope[i] > 0.3
        hma_1w_strong_down = hma_1w_slope[i] < -0.3
        
        # === INTERMEDIATE TREND (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_1d_up = hma_1d_slope[i] > 0.0
        hma_1d_down = hma_1d_slope[i] < 0.0
        
        # === KAMA SIGNAL ===
        kama_prev = kama_12h[i-1] if i > 0 else kama_12h[i]
        kama_cross_long = close[i] > kama_12h[i] and close[i-1] <= kama_prev
        kama_cross_short = close[i] < kama_12h[i] and close[i-1] >= kama_prev
        
        # Price above/below KAMA (hold condition)
        price_above_kama = close[i] > kama_12h[i]
        price_below_kama = close[i] < kama_12h[i]
        
        # === RSI FILTER (avoid extremes) ===
        rsi_ok_long = rsi_14[i] < 75.0  # not overbought
        rsi_ok_short = rsi_14[i] > 25.0  # not oversold
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 1w trend up + 1d trend up + KAMA cross long + RSI OK
        if price_above_hma_1w and price_above_hma_1d:
            if kama_cross_long and rsi_ok_long:
                if hma_1w_strong_up and hma_1d_up:
                    new_signal = POSITION_SIZE_MAX
                else:
                    new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # Require: 1w trend down + 1d trend down + KAMA cross short + RSI OK
        if price_below_hma_1w and price_below_hma_1d:
            if kama_cross_short and rsi_ok_short:
                if hma_1w_strong_down and hma_1d_down:
                    new_signal = -POSITION_SIZE_MAX
                else:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold long if still above KAMA and HTF trends intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if price_above_kama and price_above_hma_1w and price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if price_below_kama and price_below_hma_1w and price_below_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON HTF TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w:
                new_signal = 0.0
        
        # === EXIT ON KAMA CROSS BACK ===
        if in_position and position_side > 0:
            if kama_cross_short:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_cross_long:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = np.inf
        
        signals[i] = new_signal
    
    return signals