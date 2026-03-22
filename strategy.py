#!/usr/bin/env python3
"""
Experiment #112: 4h Volatility Spike Mean Reversion + 1d HMA Trend Filter

Hypothesis: Building on market analysis showing vol spike reversion works well for BTC/ETH.
This strategy captures panic/recovery cycles that destroyed simple trend strategies in 2022.

Key components:
- ATR Ratio (ATR7/ATR30) > 2.0 signals volatility spike (panic/euphoria extreme)
- Bollinger Bands (20, 2.5σ) identify price extremes during vol spikes
- 1d HMA(21) provides higher-timeframe trend bias (avoid counter-trend in major moves)
- RSI(7) confirms oversold/overbought conditions during spikes
- Asymmetric logic: long vol spikes below BB in uptrend, short vol spikes above BB in downtrend

Why this might work where others failed:
- Vol spike reversion captured the 2022 crash recovery (trend strategies got whipsawed)
- BB 2.5σ is wider than standard 2.0σ, reducing false signals
- 1d HMA filter prevents fighting major trends (unlike pure mean reversion)
- RSI confirmation adds momentum filter to avoid catching falling knives
- 4h timeframe balances signal frequency vs noise (fewer trades than 15m/30m failures)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.35 strong (discrete levels)
Stoploss: 2.5 * ATR(14) trailing

Market Analysis Insight:
"VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long.
Captures 'vol crush' after panic. Exit when ATR ratio < 1.2."
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_1d_hma_bb_rsi_v2"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, lower, sma

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period."""
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    rsi_7 = calculate_rsi(close, 7)
    
    # ATR Ratio for volatility spike detection
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    mask = atr_30 > 0
    atr_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_ratio[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0
        vol_normal = atr_ratio[i] < 1.2
        
        # === PRICE POSITION RELATIVE TO BOLLINGER BANDS ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        price_mid = bb_mid[i]
        
        # === HIGHER TIMEFRAME TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_7[i] < 30
        rsi_overbought = rsi_7[i] > 70
        rsi_neutral = 30 <= rsi_7[i] <= 70
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: Vol spike + price below BB + 1d bullish + RSI oversold
        if vol_spike and price_below_bb and bull_trend_1d and rsi_oversold:
            new_signal = SIZE_STRONG
        # Moderate: Vol spike + price below BB + 1d bullish
        elif vol_spike and price_below_bb and bull_trend_1d:
            new_signal = SIZE_BASE
        # Weak: Vol spike + price below BB + RSI oversold (ensure trades on all symbols)
        elif vol_spike and price_below_bb and rsi_oversold:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: Vol spike + price above BB + 1d bearish + RSI overbought
        if vol_spike and price_above_bb and bear_trend_1d and rsi_overbought:
            new_signal = -SIZE_STRONG
        # Moderate: Vol spike + price above BB + 1d bearish
        elif vol_spike and price_above_bb and bear_trend_1d:
            new_signal = -SIZE_BASE
        # Weak: Vol spike + price above BB + RSI overbought (ensure trades on all symbols)
        elif vol_spike and price_above_bb and rsi_overbought:
            new_signal = -SIZE_BASE
        
        # === EXIT CONDITIONS (vol normalization) ===
        # Exit long when vol normalizes and price returns to mid
        if in_position and position_side > 0 and vol_normal and close[i] > price_mid:
            new_signal = 0.0
        
        # Exit short when vol normalizes and price returns to mid
        if in_position and position_side < 0 and vol_normal and close[i] < price_mid:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr_14[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr_14[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals