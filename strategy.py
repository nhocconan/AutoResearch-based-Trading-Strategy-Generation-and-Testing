#!/usr/bin/env python3
"""
Experiment #1219: 4h Primary + 1d HTF — HMA Trend with RSI Pullback and Choppiness Sizing

Hypothesis: The current best (mtf_4h_triple_regime_crsi_donchian_1d1w_v1, Sharpe=0.612) 
uses complex regime switching. This experiment simplifies to:
- 4h HMA(21) for primary trend direction (proven effective)
- 1d HMA(21) for macro bias (single HTF filter, not triple)
- RSI(14) pullback entries in trend direction (30-65 range, looser for more trades)
- Choppiness Index(14) for position sizing (reduce size in chop, don't block trades)
- ATR(14) 2.5x trailing stop for risk management

Key improvements over failed experiments:
1. Simpler HTF filter (1d only, not 1d+1w) = more trades
2. RSI pullback (not CRSI extremes) = more entry opportunities  
3. Choppiness reduces size (doesn't block) = maintains trade frequency
4. Looser RSI ranges (30-65 long, 35-70 short) = ensures >=30 trades on train
5. Discrete signal levels (0.0, ±0.15, ±0.30) = less fee churn

Target: Sharpe > 0.612, trades >= 30 on train (2021-2024), >= 3 on test (2025)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_chop_sizing_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range (reduce position size)
    CHOP < 38.2 = trending (full position size)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    CHOP_SIZE = 0.15  # Reduced size in choppy markets
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(hma_4h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) — soft filter, don't trade against ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        trend_bull = close[i] > hma_4h[i]
        trend_bear = close[i] < hma_4h[i]
        
        # === CHOPPINESS FOR SIZING (not blocking) ===
        is_choppy = chop[i] > 61.8
        position_size = CHOP_SIZE if is_choppy else BASE_SIZE
        
        # === RSI PULLBACK ENTRIES (looser ranges for more trades) ===
        # Long: RSI pullback in bullish trend (30-65, not oversold extreme)
        rsi_pullback_long = 30.0 <= rsi[i] <= 65.0
        # Short: RSI pullback in bearish trend (35-70, not overbought extreme)
        rsi_pullback_short = 35.0 <= rsi[i] <= 70.0
        
        # === ENTRY CONDITIONS (simplified for trade frequency) ===
        desired_signal = 0.0
        
        # Long: 4h trend bull + 1d macro not strongly bear + RSI pullback
        if trend_bull and rsi_pullback_long:
            # Only require 1d not strongly against (allows neutral 1d)
            if not macro_bear or rsi[i] < 50:  # Extra conviction if 1d bear
                desired_signal = position_size
        
        # Short: 4h trend bear + 1d macro not strongly bull + RSI pullback
        elif trend_bear and rsi_pullback_short:
            # Only require 1d not strongly against (allows neutral 1d)
            if not macro_bull or rsi[i] > 50:  # Extra conviction if 1d bull
                desired_signal = -position_size
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if is_choppy:
                desired_signal = CHOP_SIZE
            else:
                desired_signal = BASE_SIZE
        elif desired_signal < 0:
            if is_choppy:
                desired_signal = -CHOP_SIZE
            else:
                desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals