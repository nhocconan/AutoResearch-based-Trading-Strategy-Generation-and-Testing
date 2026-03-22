#!/usr/bin/env python3
"""
Experiment #448: 4h HMA Trend + RSI Pullback with Daily Bias

Hypothesis: After 447 experiments, the pattern is clear - complex ensembles fail
because they're too strict (0 trades) or contradictory. This strategy SIMPLIFIES:

1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 1d HMA
   - Short bias when price < 1d HMA
   - HMA smoother than EMA, reduces whipsaw

2. RSI(14) PULLBACK ENTRIES:
   - Long: RSI < 40 in bull trend (pullback entry)
   - Short: RSI > 60 in bear trend (rally entry)
   - Looser than 30/70 to ensure sufficient trades

3. VOLUME CONFIRMATION:
   - Taker buy volume ratio > 0.55 for longs (buying pressure)
   - Taker buy volume ratio < 0.45 for shorts (selling pressure)
   - Prevents entries on low-volume fakeouts

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for crash protection

5. POSITION SIZING: 0.30 discrete
   - 30% capital per position
   - Discrete levels minimize fee churn

Why this should work on 4h:
- Simpler than failed ensembles (#436, #441, #442, #447)
- 4h captures multi-day swings without 1d slowness
- Volume filter adds edge without over-filtering
- Should generate 20-50 trades/year per symbol

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_trend_rsi_pullback_volume_atr_v1"
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
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
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
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            volume_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            volume_ratio[i] = 0.5
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        # === DAILY HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK SIGNALS ===
        rsi_long = rsi[i] < 40  # Pullback in bull trend
        rsi_short = rsi[i] > 60  # Rally in bear trend
        
        # === VOLUME CONFIRMATION ===
        volume_buying = volume_ratio[i] > 0.55  # Buying pressure
        volume_selling = volume_ratio[i] < 0.45  # Selling pressure
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Long entry: bull trend + RSI pullback + volume confirmation
        if rsi_long and bull_trend_1d and volume_buying:
            new_signal = SIZE
        
        # Short entry: bear trend + RSI rally + volume confirmation
        elif rsi_short and bear_trend_1d and volume_selling:
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
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
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