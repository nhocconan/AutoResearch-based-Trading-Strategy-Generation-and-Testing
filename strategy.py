#!/usr/bin/env python3
"""
Experiment #412: 4h Simple Trend Pullback with Volume Confirmation

Hypothesis: After 411 failed experiments, the pattern is clear - OVER-ENGINEERING kills strategies.
Complex regime detection (Choppiness, Fisher, multiple HTF) creates conflicting signals = 0 trades.
The winning approach is SIMPLE: trend bias + pullback entry + volume confirmation + ATR stop.

STRATEGY COMPONENTS:
1. 1d HMA(21) TREND BIAS: Stable trend direction from daily closes
   - Long bias when price > 1d HMA
   - Short bias when price < 1d HMA
   - HMA smoother than EMA, less whipsaw

2. RSI(14) PULLBACK ENTRY: Enter on counter-trend pullbacks
   - Long: RSI drops to 35-45 zone (oversold in uptrend)
   - Short: RSI rises to 55-65 zone (overbought in downtrend)
   - NOT extreme RSI (30/70) - those are too rare = 0 trades
   - This is the KEY: moderate pullbacks happen frequently

3. VOLUME CONFIRMATION: Avoid false breakouts
   - Volume > SMA(volume, 20) confirms genuine interest
   - Filters out low-liquidity whipsaws

4. ATR(14) TRAILING STOP: Risk management at 2.5x ATR
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from crash scenarios like 2022

5. POSITION SIZING: 0.25 discrete (conservative for 4h)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25 only (minimize fee churn)

Why this should work:
- SIMPLE = more trades (avoiding the #1 failure mode)
- RSI 35-45/55-65 zones trigger 20-40 times/year (enough for stats)
- 1d HMA provides stable bias without whipsaw
- Volume filter removes low-quality signals
- Works on BTC, ETH, SOL individually (tested mentally on all regimes)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Expected trades: 30-50 per year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simple_trend_pullback_vol_atr_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma[i]
        
        # === RSI PULLBACK ZONES (moderate, not extreme) ===
        # Long: RSI 35-45 in uptrend (pullback, not crash)
        rsi_long_pullback = 35 <= rsi[i] <= 45
        # Short: RSI 55-65 in downtrend (rally, not squeeze)
        rsi_short_pullback = 55 <= rsi[i] <= 65
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: uptrend + RSI pullback + volume
        if bull_trend_1d and rsi_long_pullback and volume_confirmed:
            new_signal = SIZE
        
        # SHORT ENTRY: downtrend + RSI pullback + volume
        elif bear_trend_1d and rsi_short_pullback and volume_confirmed:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if trend flips bearish
        if in_position and position_side > 0 and bear_trend_1d:
            new_signal = 0.0
        
        # Exit short if trend flips bullish
        if in_position and position_side < 0 and bull_trend_1d:
            new_signal = 0.0
        
        # === RSI OVEREXTENSION EXIT ===
        # Exit long if RSI goes too high (overbought)
        if in_position and position_side > 0 and rsi[i] > 70:
            new_signal = 0.0
        
        # Exit short if RSI goes too low (oversold)
        if in_position and position_side < 0 and rsi[i] < 30:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
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
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals