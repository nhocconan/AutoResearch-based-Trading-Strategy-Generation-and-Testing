#!/usr/bin/env python3
"""
Experiment #408: 30m Primary + 4h/1d HTF — Simplified Trend + RSI Pullback

Hypothesis: Lower timeframe strategies fail due to TOO MANY confluence filters.
This strategy uses SIMPLIFIED entry logic to ensure trades actually occur.

Key innovations vs failed 30m strategies (#398, #405 with Sharpe=0.000):
1. FEWER entry conditions - only 2-3 filters max (not 5+)
2. 4h HMA for trend direction (proven in winning strategies)
3. 30m RSI(7) pullback entries (simple, frequent enough)
4. 1d HMA for overall bias (loose filter, not strict)
5. Position size 0.25 (conservative for 30m, target 40-70 trades/year)
6. Asymmetric stoploss: 2.5x ATR longs, 2.0x ATR shorts
7. NO session filter (killed trade frequency in #398/#405)
8. NO volume filter (too restrictive for 30m)

Why this should beat Sharpe=0.000 lower TF failures:
- Simpler entry = more trades (critical for lower TF)
- HTF trend filter prevents whipsaw (4h HMA proven edge)
- RSI(7) pullback = frequent entry signals in trend
- Conservative sizing (0.25) controls drawdown
- Target: 40-70 trades/year on 30m (within 30-80 limit)

Target: Sharpe > 0.3, 40-70 trades/year, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA."""
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    ema_8 = calculate_ema(close, period=8)
    ema_21 = calculate_ema(close, period=21)
    
    # Calculate and align HTF HMA for trend (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol normalization
    atr_median = np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 30m (target 40-70 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]):
            continue
        
        # === HTF TREND (4h HMA) ===
        # Price above 4h HMA = bullish bias
        # Price below 4h HMA = bearish bias
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === HTF BIAS (1d HMA) - Loose filter ===
        # Only used to reduce position size against major trend
        bias_bullish = close[i] > hma_1d_aligned[i]
        bias_bearish = close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI(7) < 35 in uptrend (pullback entry)
        # Short: RSI(7) > 65 in downtrend (rally entry)
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        
        # === EMA CROSS CONFIRMATION ===
        # Long: EMA(8) > EMA(21)
        # Short: EMA(8) < EMA(21)
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        
        # === VOL FILTER (loose) ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 3.0:
            position_size = BASE_SIZE * 0.5  # Reduce in extreme vol
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Simpler conditions for more trades
        if trend_bullish:
            # Path 1: RSI pullback in uptrend (primary entry)
            if rsi_oversold:
                desired_signal = position_size
            # Path 2: EMA bullish + RSI not overbought
            elif ema_bullish and rsi_14[i] < 70.0:
                desired_signal = position_size * 0.5  # Smaller position
        
        # SHORT SETUP - Simpler conditions for more trades
        if trend_bearish:
            # Path 1: RSI rally in downtrend (primary entry)
            if rsi_overbought:
                desired_signal = -position_size
            # Path 2: EMA bearish + RSI not oversold
            elif ema_bearish and rsi_14[i] > 30.0:
                desired_signal = -position_size * 0.5  # Smaller position
        
        # === Bias conflict reduction ===
        # Reduce position if 4h and 1d disagree
        if desired_signal > 0 and bias_bearish:
            desired_signal = desired_signal * 0.5
        if desired_signal < 0 and bias_bullish:
            desired_signal = desired_signal * 0.5
        
        # === STOPLOSS CHECK (Asymmetric) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXIT (extreme reached) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF reversal) ===
        if in_position and position_side > 0 and trend_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and trend_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and trend_bullish:
                desired_signal = position_size
            elif position_side < 0 and trend_bearish:
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals