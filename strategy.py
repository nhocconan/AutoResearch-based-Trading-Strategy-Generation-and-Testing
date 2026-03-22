#!/usr/bin/env python3
"""
Experiment #009: 4h Primary + 1d HTF — Simplified Trend + Mean Reversion

Hypothesis: Previous strategies failed due to over-filtering (too many confluence
requirements = 0 trades). This strategy uses SIMPLIFIED logic:

1. 1d HMA(21) for major trend bias (price vs HMA only)
2. 4h RSI(14) for entry timing (extreme <35 or >65 — loosened from Connors)
3. Bollinger Band position for confirmation (price outside bands)
4. ATR(14) volatility filter (skip if ATR ratio extreme)
5. Simple 2.5x ATR trailing stoploss

Why this should work:
- Fewer filters = more trades (target 20-50/year on 4h)
- RSI(14) extremes have proven mean-reversion edge
- BB position confirms overextension
- 1d HMA provides trend bias without over-complicating
- Discrete sizing (0.25/0.30) minimizes fee churn

Timeframe: 4h (REQUIRED per experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_bb_hma_1d_v1"
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
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Bandwidth position: -1 = at lower, 0 = at middle, +1 = at upper
    bb_position = (close - lower) / (upper - lower + 1e-10)
    bb_position = np.nan_to_num(bb_position, nan=0.5)
    bb_position = np.clip(bb_position, 0, 1)
    return sma, upper, lower, bb_position

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr_ratio(atr_short, atr_long):
    """Calculate ATR ratio for volatility filter."""
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    rsi_14 = calculate_rsi(close, 14)
    bb_sma, bb_upper, bb_lower, bb_position = calculate_bollinger_bands(close, 20, 2.0)
    
    # ATR ratio for volatility filter
    atr_ratio = calculate_atr_ratio(atr_7, atr_14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
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
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_position[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        # RSI < 35 = oversold (long opportunity)
        # RSI > 65 = overbought (short opportunity)
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === BOLLINGER BAND POSITION ===
        # bb_position < 0.1 = near lower band (long)
        # bb_position > 0.9 = near upper band (short)
        bb_lower_extreme = bb_position[i] < 0.15
        bb_upper_extreme = bb_position[i] > 0.85
        
        # === VOLATILITY FILTER ===
        # Skip if ATR ratio extreme (>2.5 = vol spike, <0.5 = vol crush)
        vol_normal = 0.5 < atr_ratio[i] < 2.5
        
        # === ENTRY LOGIC (SIMPLIFIED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 1d bullish + RSI oversold + BB lower extreme
        if trend_1d_bullish and rsi_oversold and bb_lower_extreme and vol_normal:
            new_signal = BASE_SIZE
        
        # SHORT: 1d bearish + RSI overbought + BB upper extreme
        if trend_1d_bearish and rsi_overbought and bb_upper_extreme and vol_normal:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~50 days on 4h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 40:
                new_signal = BASE_SIZE * 0.7
            elif trend_1d_bearish and rsi_14[i] > 60:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
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
                # Position flip
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