#!/usr/bin/env python3
"""
Experiment #245: 1h Primary + 4h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 8 consecutive failures with complex regime switching (Chop+CRSI+session),
return to proven HMA+RSI framework from #236 but on 1h timeframe. Key lesson: TOO MANY
filters = 0 trades (experiments #238, #240, #242 all had Sharpe=0.000 = no trades).

Design simplification:
1. 4h HMA(16/48) for trend direction (proven in #236)
2. 1h RSI(14) pullback to 30-70 zone (wider to ensure trades)
3. 1d HMA(21) for macro bias alignment
4. ATR(14) 2.5x trailing stoploss
5. NO session filter, NO volume filter, NO Choppiness (these killed trade frequency)
6. Position size: 0.25 (smaller for 1h volatility vs 12h)

TARGET: 40-80 trades/year on 1h, Sharpe > 0.5 on ALL symbols
CRITICAL: Entry conditions MUST be loose enough to generate trades (>30 train, >3 test)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate 4h HMA for trend (aligned properly)
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, 48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        macro_bullish = price_above_hma_1d
        macro_bearish = price_below_hma_1d
        
        # === 4h TREND DETECTION (HMA crossover) ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 1h TREND DETECTION (HMA crossover) ===
        hma_1h_bullish = hma_16[i] > hma_48[i]
        hma_1h_bearish = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK ZONES (wider to ensure trades) ===
        # Long: RSI pulled back but not oversold (30-65)
        rsi_long_pullback = 30.0 <= rsi_14[i] <= 65.0
        # Short: RSI rallied but not overbought (35-70)
        rsi_short_pullback = 35.0 <= rsi_14[i] <= 70.0
        rsi_overbought = rsi_14[i] > 75.0
        rsi_oversold = rsi_14[i] < 25.0
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h trend bullish + 1h pullback + RSI in zone
        if hma_4h_bullish and hma_1h_bullish and rsi_long_pullback and not rsi_overbought:
            if macro_bullish:
                desired_signal = POSITION_SIZE_FULL
            else:
                desired_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: 4h trend bearish + 1h pullback + RSI in zone
        elif hma_4h_bearish and hma_1h_bearish and rsi_short_pullback and not rsi_oversold:
            if macro_bearish:
                desired_signal = -POSITION_SIZE_FULL
            else:
                desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === MACRO REVERSAL EXIT (strong contra signal) ===
        if in_position and position_side > 0 and macro_bearish and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bullish and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if trend still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and hma_4h_bullish and rsi_14[i] < 80.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and hma_4h_bearish and rsi_14[i] > 20.0:
                desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals