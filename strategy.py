#!/usr/bin/env python3
"""
Experiment #015: 1h Primary + 4h/1d HTF — Simplified Mean Reversion in Trend

Hypothesis: Previous strategies failed due to OVER-FILTERING (too many confluence
conditions = 0 trades) or UNDER-FILTERING (too many trades = fee drag).

This strategy uses SIMPLIFIED but ROBUST confluence:
1. 1d HMA(21) = MAJOR trend bias (only trade WITH this trend)
2. 4h HMA(21) = INTERMEDIATE confirmation (must align with 1d)
3. 1h RSI(14) < 40 or > 60 = PULLBACK entry (not extreme, more frequent)
4. 1h Bollinger Band touch = ENTRY trigger (price at band edge)
5. ATR(14) volatility filter = avoid low-vol whipsaws
6. 2.5x ATR trailing stoploss

Why this should work:
- Fewer filters = more trades (target 40-80/year, not 0)
- Still multi-timeframe = HTF frequency with LTF precision
- RSI 40/60 (not 30/70) = catches more pullbacks in strong trends
- BB touch confirms entry timing within the pullback
- Works in both bull AND bear markets (directional based on 1d HMA)
- Discrete sizing (0.25) controls drawdown during 2022-style crashes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 (conservative for 1h TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol

Key difference from failed #014: SIMPLER entry conditions, no session filter
(24/7 crypto), no volume filter (crypto volume unreliable), no Choppiness
(too many calculations, marginal benefit).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_simplified_mr_4h1d_hma_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands with configurable multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # ATR average for volatility filter
    atr_avg_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        if np.isnan(atr_avg_50[i]) or atr_avg_50[i] == 0:
            continue
        
        # === VOLATILITY FILTER ===
        # Avoid low-volatility periods (whipsaw prone)
        vol_ok = atr_14[i] > 0.8 * atr_avg_50[i]
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === RSI PULLBACK CONDITIONS ===
        # RSI < 40 = pullback in uptrend (long opportunity)
        # RSI > 60 = pullback in downtrend (short opportunity)
        # Using 40/60 instead of 30/70 for MORE trades
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === BOLLINGER BAND TOUCH ===
        # Price at or beyond band = entry trigger
        bb_long_trigger = close[i] <= bb_lower[i] * 1.001  # at or below lower band
        bb_short_trigger = close[i] >= bb_upper[i] * 0.999  # at or above upper band
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 1d bullish + 4h bullish + RSI pullback + BB touch + vol ok
        if trend_1d_bullish and trend_4h_bullish:
            if rsi_oversold and bb_long_trigger and vol_ok:
                new_signal = current_size
        
        # SHORT ENTRIES
        # Require: 1d bearish + 4h bearish + RSI pullback + BB touch + vol ok
        if trend_1d_bearish and trend_4h_bearish:
            if rsi_overbought and bb_short_trigger and vol_ok:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~6 days on 1h), allow weaker entry
        # This ensures minimum trade count without over-trading
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_4h_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.6
            elif trend_1d_bearish and trend_4h_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        # Exit if major trend reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and rsi_14[i] > 60:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and rsi_14[i] < 40:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
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