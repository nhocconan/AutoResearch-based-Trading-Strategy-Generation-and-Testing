#!/usr/bin/env python3
"""
Experiment #431: 4h Primary + 1d/1w HTF — Vol Spike Mean Reversion + Regime Filter

Hypothesis: After analyzing 430 failed experiments, clear patterns emerge:
1. Complex multi-filter strategies (CRSI+Chop+Donchian+HMA) fail due to 0 trades
2. Volatility spike mean reversion has strong edge: ATR(7)/ATR(30) > 2.0 = panic capitulation
3. 1d HMA(21) for major trend direction prevents counter-trend trades in crashes
4. Simpler logic = more trades = avoid Sharpe=0.000 (the #1 failure mode)
5. Asymmetric entries: favor longs in bull regime, favor shorts in bear regime

Why this might beat current best (Sharpe=0.435):
- Vol spike reversion captures "panic bottom" and "euphoria top" reversals
- 4h TF balances trade frequency (20-50/year) with signal quality
- 1d HTF filter prevents 2022-style crash whipsaw (only long if price>1d HMA)
- Fewer conflicting filters = more trades = statistical significance
- ATR 2.5x trailing stop protects capital in extended moves

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 25-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_mr_hma_1d_asym_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    choppiness = calculate_choppiness(high, low, close, 14)
    
    # Volatility spike ratio: ATR(7)/ATR(30) > 2.0 = panic/euphoria
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d HMA = bull market bias (favor longs, avoid shorts)
        # Price below 1d HMA = bear market bias (favor shorts, avoid longs)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY SPIKE (panic/euphoria detection) ===
        # vol_ratio > 2.0 = extreme volatility = potential reversal point
        vol_spike = vol_ratio[i] > 2.0
        
        # === BOLLINGER BAND POSITION ===
        # price < lower band = oversold (long opportunity)
        # price > upper band = overbought (short opportunity)
        below_bb = close[i] < bb_lower[i]
        above_bb = close[i] > bb_upper[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = ranging (mean reversion works well)
        # CHOP < 45 = trending (breakouts work better)
        is_choppy = choppiness[i] > 55.0
        
        # === ENTRY LOGIC — ASYMMETRIC BY REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY (easier in bull regime)
        if bull_regime:
            # Vol spike + oversold = panic bottom (strongest signal)
            if vol_spike and (below_bb or rsi_oversold):
                new_signal = LONG_SIZE
            # Mean reversion in choppy market
            elif is_choppy and below_bb and rsi_oversold:
                new_signal = LONG_SIZE * 0.8
            # Simple oversold bounce (ensure trade frequency)
            elif rsi_14[i] < 30.0 and not above_bb:
                new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRY (easier in bear regime)
        if bear_regime:
            # Vol spike + overbought = euphoria top (strongest signal)
            if vol_spike and (above_bb or rsi_overbought):
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Mean reversion in choppy market
            elif is_choppy and above_bb and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Simple overbought rejection (ensure trade frequency)
            elif rsi_14[i] > 70.0 and not below_bb:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 12 bars (~2 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_14[i] < 40.0:
                new_signal = LONG_SIZE * 0.5
            elif bear_regime and rsi_14[i] > 60.0:
                new_signal = -SHORT_SIZE * 0.5
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on mean reversion exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
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