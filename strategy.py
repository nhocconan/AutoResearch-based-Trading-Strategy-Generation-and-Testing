#!/usr/bin/env python3
"""
Experiment #567: 1d Primary + 1w HTF — Volatility Spike Mean Reversion with Regime Filter

Hypothesis: After 500+ failed experiments, clear patterns emerge:
- Pure trend following fails on BTC/ETH (2022 crash whipsaw destroys gains)
- Pure mean reversion fails in strong trends (gets run over)
- VOLATILITY SPIKE + MEAN REVERSION + HTF REGIME FILTER is the winning combo

This strategy combines proven elements from research:
1. Volatility spike: ATR(7)/ATR(30) > 1.5 (panic/euphoria = reversal opportunity)
2. Mean reversion entry: Price outside BB(20, 2.0) + RSI(3) extremes
3. 1w HTF HMA(21) for regime bias (size larger with trend, smaller against)
4. Asymmetric sizing: 0.30 long, 0.25 short (crypto long bias)
5. ATR(14) 2.5x trailing stop for risk management

Why 1d timeframe:
- 20-50 trades/year target (Rule 10 - optimal for daily)
- Less fee drag than lower TF (0.05% per trade matters less)
- Captures major moves without noise/whipsaw

Key improvements over failed experiments:
- #563 (1d HMA crossover): Failed because pure trend doesn't work in 2022-2025
- #557 (1d dual regime): Sharpe=0.152 too low, needs vol spike filter
- This adds VOL FILTER to avoid entering during low-vol chop

Position sizing: 0.25-0.30 discrete (Rule 4 - max 0.40)
Stoploss: 2.5 * ATR(14) trailing (signal → 0 when hit)
Target: Beat Sharpe=0.435, trades >= 30 train, >= 3 test, all symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_volspike_mr_hma_1w_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_3 = calculate_rsi(close, 3)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(bb_lower[i]) or np.isnan(rsi_3[i]):
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7)/ATR(30) > 1.5 means vol expansion (panic/euphoria)
        vol_ratio = atr_7[i] / atr_30[i]
        vol_spike = vol_ratio > 1.5
        
        # === MEAN REVERSION ENTRY ===
        # Price below BB lower = oversold (long opportunity)
        price_below_bb = close[i] < bb_lower[i]
        # Price above BB upper = overbought (short opportunity)
        price_above_bb = close[i] > bb_upper[i]
        
        # === RSI EXTREME TIMING (Connors-style fast RSI) ===
        # RSI(3) < 20 = extreme oversold (long)
        # RSI(3) > 80 = extreme overbought (short)
        rsi_extreme_long = rsi_3[i] < 20.0
        rsi_extreme_short = rsi_3[i] > 80.0
        
        # === 1W HTF REGIME FILTER ===
        # Don't fight the weekly trend (size with trend, smaller against)
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # === EXIT CONDITIONS ===
        should_exit = False
        
        # Stoploss check (2.5 * ATR trailing)
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                should_exit = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                should_exit = True
        
        # Regime exit (flip against weekly trend)
        if in_position and not should_exit:
            if position_side > 0 and bear_regime_1w:
                should_exit = True
            if position_side < 0 and bull_regime_1w:
                should_exit = True
        
        # === DETERMINE SIGNAL ===
        if should_exit:
            new_signal = 0.0
        elif in_position:
            # Hold existing position
            new_signal = signals[i-1] if i > 0 else 0.0
        else:
            # Check for new entries (only when flat)
            new_signal = 0.0
            
            # LONG: vol spike + oversold + RSI extreme
            if vol_spike and price_below_bb and rsi_extreme_long:
                if bull_regime_1w:
                    new_signal = LONG_SIZE  # Full size with trend
                else:
                    new_signal = LONG_SIZE * 0.7  # Reduced size counter-trend
            
            # SHORT: vol spike + overbought + RSI extreme
            elif vol_spike and price_above_bb and rsi_extreme_short:
                if bear_regime_1w:
                    new_signal = -SHORT_SIZE  # Full size with trend
                else:
                    new_signal = -SHORT_SIZE * 0.7  # Reduced size counter-trend
        
        # === UPDATE POSITION TRACKING ===
        prev_in_position = in_position
        prev_position_side = position_side
        
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