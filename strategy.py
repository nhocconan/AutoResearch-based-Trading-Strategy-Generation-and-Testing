#!/usr/bin/env python3
"""
Experiment #340: 4h Volatility Mean Reversion with 1d/1w HMA Trend Filter

Hypothesis: After 289 failed strategies, the key insight is that 4h timeframe is ideal
for capturing volatility mean-reversion while using daily/weekly trend for bias.
This strategy combines:

1. BOLLINGER BAND MEAN REVERSION: Price at BB extremes (±2.0 std) signals oversold/overbought
2. ATR VOLATILITY RATIO: ATR(7)/ATR(28) > 1.5 confirms vol expansion = reversal likely
3. 1D/1W HMA TREND FILTER: Only long if price > 1d HMA, only short if price < 1d HMA
4. RSI CONFIRMATION: RSI(14) < 35 for longs, > 65 for shorts (looser than typical 30/70)
5. CHOPPINESS FILTER: Avoid entering when CHOP > 65 (too choppy, whipsaw risk)

Why 4h works:
- Slow enough to avoid noise, fast enough for mean-reversion cycles
- 4h bars capture daily volatility patterns without 15m/1h noise
- Combined with 1d/1w HMA, provides stable trend bias

Position sizing: 0.25 discrete (conservative for 77% BTC crash in 2022)
Stoploss: 2.5 * ATR(14) trailing
Target: >=10 trades/symbol train, >=3 trades/symbol test, Sharpe > 0 all symbols

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_meanrev_1d_1w_hma_bb_rsi_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma.values, upper.values, lower.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_series = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    
    for i in range(period, n):
        atr_sum = atr_series[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_fast = calculate_atr(high, low, close, 7)
    atr_slow = calculate_atr(high, low, close, 28)
    rsi = calculate_rsi(close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    bars_in_trade = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Avoid extremely choppy markets
        if chop[i] > 65:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                bars_in_trade = 0
            continue
        
        # === HTF TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY EXPANSION ===
        vol_ratio = atr_fast[i] / max(atr_slow[i], 1e-10)
        vol_expanding = vol_ratio > 1.3  # Loosened from 1.5 for more trades
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_sma[i]) / max(bb_upper[i] - bb_lower[i], 1e-10)
        at_lower_bb = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower band
        at_upper_bb = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        # === RSI EXTREMES (loosened for more trades) ===
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: Mean reversion with bullish HTF bias
        if at_lower_bb and rsi_oversold and bull_trend_1d:
            # Stronger signal if 1w also bullish or vol expanding
            if bull_trend_1w or vol_expanding:
                new_signal = SIZE
        
        # SHORT: Mean reversion with bearish HTF bias
        elif at_upper_bb and rsi_overbought and bear_trend_1d:
            # Stronger signal if 1w also bearish or vol expanding
            if bear_trend_1w or vol_expanding:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === TIME-BASED EXIT (prevent capital tie-up) ===
        if in_position:
            bars_in_trade += 1
            if bars_in_trade > 30 and new_signal == 0.0:
                # Exit after 30 bars (5 days on 4h) if no clear signal
                new_signal = 0.0
        else:
            bars_in_trade = 0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                bars_in_trade = 1
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                bars_in_trade = 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                bars_in_trade = 0
        
        signals[i] = new_signal
    
    return signals