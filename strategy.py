#!/usr/bin/env python3
"""
Experiment #391: 15m RSI Pullback + 1h HMA Trend Bias + ATR Stoploss

Hypothesis: After 390 failed experiments, the pattern is clear - OVER-FILTERING kills strategies.
Complex regime detection (CHOP + ADX + multiple HTFs) results in 0-5 trades. 
For 15m timeframe, we need SIMPLER logic that actually generates trades.

STRATEGY COMPONENTS:
1. 1h HMA(21) TREND BIAS: Single HTF filter for directional bias
   - Long only when price > 1h HMA (bullish HTF)
   - Short only when price < 1h HMA (bearish HTF)
   - HMA smoother than EMA, less whipsaw on 1h

2. 15m RSI(14) PULLBACK: Entry trigger on pullbacks within trend
   - Long: RSI < 35 (oversold pullback in uptrend)
   - Short: RSI > 65 (overbought pullback in downtrend)
   - This generates 50-100 trades/year per symbol

3. ATR(14) TRAILING STOP: Risk management
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from adverse moves

4. POSITION SIZING: 0.25 discrete (conservative for 15m volatility)
   - Max 25% capital per position
   - Discrete levels (0.0, ±0.25) minimize fee churn

Why this should work on 15m:
- Simple enough to generate trades (not over-filtered like exp 379-390)
- HTF bias prevents counter-trend trades (major failure mode)
- RSI pullback entries have proven edge in crypto
- Should work on BTC, ETH, SOL individually (not SOL-biased)
- 15m captures intraday moves that 4h/1d miss

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_1h_hma_atr_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
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
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === 1h HMA TREND BIAS ===
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        
        # === 15m RSI PULLBACK SIGNALS ===
        # Long: RSI oversold (<35) in uptrend
        rsi_oversold = rsi[i] < 35
        # Short: RSI overbought (>65) in downtrend
        rsi_overbought = rsi[i] > 65
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: RSI pullback in uptrend
        if bull_trend_1h and rsi_oversold:
            new_signal = SIZE
        
        # SHORT: RSI pullback in downtrend
        elif bear_trend_1h and rsi_overbought:
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
        # Exit long if trend turns bearish
        if in_position and position_side > 0 and bear_trend_1h:
            new_signal = 0.0
        
        # Exit short if trend turns bullish
        if in_position and position_side < 0 and bull_trend_1h:
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