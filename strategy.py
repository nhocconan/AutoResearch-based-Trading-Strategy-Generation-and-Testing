#!/usr/bin/env python3
"""
Experiment #607: 15m Multi-Timeframe RSI Pullback with 4h HMA Trend Filter

Hypothesis: After 538+ failures, the key lesson is that COMPLEX regime detection
(Choppiness, multiple filters) often results in 0 trades. For 15m timeframe:

1. Use SIMPLE 4h HMA(21) for trend bias (call get_htf_data ONCE before loop)
2. Use FAST 15m RSI(7) for pullback entries (faster than RSI(14) for 15m)
3. Long: 4h HMA bullish + 15m RSI dips to 30-40 zone then recovers above 35
4. Short: 4h HMA bearish + 15m RSI rises to 60-70 zone then falls below 65
5. ATR(14) stoploss at 2.0x trailing
6. Position size: 0.28 discrete (max 0.40)

Why this should work on 15m:
- 15m has more volatility = more RSI extremes = MORE trades (critical!)
- 4h HMA filter prevents trading against major trend
- RSI(7) is faster than RSI(14) = more signals on 15m
- LOOSE thresholds (30-40, 60-70) ensure we get trades
- Simple logic = fewer conditions that can all fail simultaneously

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_pullback_4h_hma_trend_atr_v1"
timeframe = "15m"
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

def calculate_rsi(close, period=7):
    """Calculate RSI (Relative Strength Index) - faster period for 15m."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Track RSI state for pullback detection
    prev_rsi = 0.0
    rsi_was_oversold = False
    rsi_was_overbought = False
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(ema_21[i]):
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === 15M RSI PULLBACK DETECTION ===
        rsi_current = rsi_7[i]
        
        # Track if RSI was in oversold/overbought zone recently
        if rsi_current < 40:
            rsi_was_oversold = True
        if rsi_current > 60:
            rsi_was_overbought = True
        
        # RSI recovery from oversold (for long entry)
        rsi_recovery_long = rsi_was_oversold and rsi_current > 35 and prev_rsi <= 35
        
        # RSI rejection from overbought (for short entry)
        rsi_rejection_short = rsi_was_overbought and rsi_current < 65 and prev_rsi >= 65
        
        # Reset flags after signal
        if rsi_recovery_long or rsi_rejection_short:
            rsi_was_oversold = False
            rsi_was_overbought = False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: 4h bullish bias + RSI pullback recovery
        if bull_bias and rsi_recovery_long:
            # Additional filter: price above 15m EMA21 for momentum
            if close[i] > ema_21[i]:
                new_signal = SIZE
        
        # Short: 4h bearish bias + RSI overbought rejection
        if bear_bias and rsi_rejection_short:
            # Additional filter: price below 15m EMA21 for momentum
            if close[i] < ema_21[i]:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and bear_bias:
                trend_reversal = True
            if position_side < 0 and bull_bias:
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
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
        prev_rsi = rsi_current
    
    return signals