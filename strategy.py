#!/usr/bin/env python3
"""
Experiment #422: 12h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: 12h timeframe with daily bias filter will produce 30-60 trades/year
with better risk-adjusted returns than complex regime-switching strategies.

Key lessons from 354 failed experiments:
1. Complex CRSI + Choppiness regimes = 0 trades (too restrictive)
2. Simple HMA + RSI with moderate thresholds = consistent trades
3. 12h TF reduces noise vs 4h while maintaining adequate frequency
4. 1d HTF bias is stronger than 4h for overall trend direction
5. Entry conditions MUST be loose enough to trigger on major moves

Strategy design:
- 1d HMA(21) for HTF bias (bull/bear filter)
- 12h HMA(21/50) crossover for trend direction
- RSI(14) pullback entries: <40 for long, >60 for short (NOT extreme 15/85)
- ATR(14) trailing stoploss at 2.5x for risk management
- Position size: 0.25-0.30 discrete levels
- Target: 40-80 trades over 4-year train, Sharpe > 0.612

Why this should beat #417 (Sharpe=0.042):
- 12h has more signals than 1d while maintaining quality
- Moderate RSI thresholds ensure trade frequency (critical!)
- Simpler logic = fewer conditions that can all fail simultaneously
- 1d HTF bias is proven to work (current best uses 1d/1w)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align HTF HMA for bias (1d and 1w)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        # Strong bullish: price above both 1d and 1w HMA
        # Strong bearish: price below both 1d and 1w HMA
        # Neutral: mixed signals
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        bullish_bias = price_above_hma_1d and price_above_hma_1w
        bearish_bias = price_below_hma_1d and price_below_hma_1w
        
        # === PRIMARY TREND (12h HMA crossover) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === SMA200 FILTER (long-term trend) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK THRESHOLDS (MODERATE - ensure trade frequency) ===
        # Use 35/65 instead of extreme 15/85 to get more trades
        rsi_oversold = rsi_14[i] < 40.0  # Long entry on pullback
        rsi_overbought = rsi_14[i] > 60.0  # Short entry on rally
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5  # Reduce size in high vol
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Multiple confluence but NOT too strict
        # Need: bullish HTF bias OR (HMA bullish + above SMA200) + RSI pullback
        long_conditions = 0
        if bullish_bias:
            long_conditions += 2  # Strong bias
        if price_above_hma_1d:
            long_conditions += 1
        if hma_bullish:
            long_conditions += 1
        if price_above_sma200:
            long_conditions += 1
        
        if long_conditions >= 2 and rsi_oversold:
            desired_signal = position_size
        elif long_conditions >= 3 and rsi_14[i] < 45.0:
            # Slightly less oversold if more confluence
            desired_signal = position_size * 0.7
        
        # SHORT SETUP
        short_conditions = 0
        if bearish_bias:
            short_conditions += 2
        if price_below_hma_1d:
            short_conditions += 1
        if hma_bearish:
            short_conditions += 1
        if price_below_sma200:
            short_conditions += 1
        
        if short_conditions >= 2 and rsi_overbought:
            desired_signal = -position_size
        elif short_conditions >= 3 and rsi_14[i] > 55.0:
            desired_signal = -position_size * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and bearish_bias:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and bullish_bias:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and not bearish_bias:
                desired_signal = position_size
            elif position_side < 0 and not bullish_bias:
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