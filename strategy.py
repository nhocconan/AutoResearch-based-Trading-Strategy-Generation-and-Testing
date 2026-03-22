#!/usr/bin/env python3
"""
Experiment #014: 4h HMA Trend + RSI Pullback with 12h/1d Confirmation

Hypothesis: After 13 failed experiments, return to proven trend-following with 
pullback entries (not breakouts). Breakout strategies (#012, #013) showed 
positive returns but negative Sharpe due to whipsaws. Pullback entries in 
established trends have higher win rates (60-70% vs 45-55% for breakouts).

Key changes from #012:
1. Primary TF = 4h (more signals than 12h, still controlled frequency)
2. RSI pullback entries instead of Donchian breakouts (better win rate)
3. Simpler confluence logic (2-3 conditions, not 5+)
4. ATR trailing stop with tighter 2.0x multiplier
5. Position sizing: 0.25 long, 0.20 short (bear market bias)

Logic:
- 4h HMA(21) vs HMA(48) = primary trend direction
- 12h HMA(21) = intermediate confirmation
- 1d HMA(21) = secular bias (only trade with major trend)
- RSI(14) pullback = entry timing (RSI 35-45 for long, 55-65 for short)
- ATR(14) trailing stop = 2.0x for protection

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.25 discrete
Stoploss: 2.0 * ATR(14) trailing
Target: 30-60 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h_1d_confirm_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_LONG = 0.25
    BASE_SIZE_SHORT = 0.20  # Smaller for shorts (bear market bias in test period)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or hma_12h_21_aligned[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or hma_1d_21_aligned[i] == 0:
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D SECULAR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H INTERMEDIATE TREND ===
        hma_12h_bullish = close[i] > hma_12h_21_aligned[i]
        hma_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_48[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_48[i]
        
        # === RSI PULLBACK CONDITIONS ===
        # Long: RSI pulled back to 35-50 in uptrend
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        # Short: RSI rallied to 50-65 in downtrend
        rsi_pullback_short = 50.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h trend + 12h confirmation + RSI pullback
        # Need: 4h HMA bullish + (12h bullish OR 1d bullish) + RSI pullback
        long_trend_ok = hma_4h_bullish and (hma_12h_bullish or daily_bullish)
        if long_trend_ok and rsi_pullback_long:
            new_signal = BASE_SIZE_LONG
        
        # SHORT ENTRY: 4h trend + 12h confirmation + RSI pullback
        # Need: 4h HMA bearish + (12h bearish OR 1d bearish) + RSI pullback
        short_trend_ok = hma_4h_bearish and (hma_12h_bearish or daily_bearish)
        if short_trend_ok and rsi_pullback_short:
            new_signal = -BASE_SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h HMA turns bearish
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            # Exit short if 4h HMA turns bullish
            if position_side < 0 and hma_4h_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals