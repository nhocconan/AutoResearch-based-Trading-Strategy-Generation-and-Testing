#!/usr/bin/env python3
"""
EXPERIMENT #016 - 1h Primary with 4h HMA Trend + RSI Pullback + ATR Stop
=========================================================================
Hypothesis: Combining 4h HMA(21) trend direction with 1h RSI(14) pullback entries
will capture trends while avoiding chasing extended moves. The 4h HMA is smoother
than EMA and reduces whipsaws. RSI extremes (30/70) in direction of trend provide
high-probability entries. ATR(14) trailing stop limits drawdown.

Key differences from failed strategies:
- Smaller position size (0.25 base vs 0.35) for better DD control
- Stricter stoploss (2*ATR vs 3*ATR) to cut losses faster
- Volume confirmation filter to avoid low-liquidity traps
- Discrete signal levels (0.0, ±0.25) to minimize fee churn
- Proper MTF alignment using mtf_data helper (ONCE before loop)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_hma_rsi_atr_stop_v2"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Array"""
    close_s = pd.Series(close)
    wma1 = close_s.rolling(window=period//2, min_periods=period//2).mean()
    wma2 = close_s.rolling(window=period, min_periods=period).mean()
    diff = 2 * wma1 - wma2
    hma = diff.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg.replace(0, np.inf)
    return (vol_ratio > threshold).astype(float)


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD 4H HTF DATA ONCE BEFORE LOOP (CRITICAL) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA(21) for trend direction
    hma_4h = calculate_hma(close_4h, 21)
    
    # Align 4h HMA to 1h timeframe (auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # === CALCULATE 1H INDICATORS ===
    # 1h RSI(14) for entry timing
    rsi_1h = calculate_rsi(close, 14)
    
    # 1h ATR(14) for stoploss
    atr_1h = calculate_atr(high, low, close, 14)
    
    # 1h volume spike filter
    vol_spike_1h = calculate_volume_spike(volume, 20, 1.5)
    
    # 1h HMA(21) for additional trend confirmation
    hma_1h = calculate_hma(close, 21)
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size - conservative for DD control
    HALF_SIZE = 0.125  # Half position for take profit
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    # Warmup period
    warmup = max(50, int(np.sqrt(21)) + 21)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            continue
        
        # === TREND FILTER: 4h HMA direction ===
        # Use slope of 4h HMA for trend (compare to 3 bars ago)
        if i >= 3 and not np.isnan(hma_4h_aligned[i-3]):
            hma_slope_4h = hma_4h_aligned[i] - hma_4h_aligned[i-3]
        else:
            hma_slope_4h = 0.0
        
        trend_bullish = hma_slope_4h > 0
        trend_bearish = hma_slope_4h < 0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h trend up + 1h RSI oversold (<35) + volume confirmation
        long_signal = (trend_bullish and 
                       rsi_1h[i] < 35 and 
                       vol_spike_1h[i] >= 0.5)
        
        # Short: 4h trend down + 1h RSI overbought (>65) + volume confirmation
        short_signal = (trend_bearish and 
                        rsi_1h[i] > 65 and 
                        vol_spike_1h[i] >= 0.5)
        
        # === EXIT CONDITIONS (opposite signal) ===
        exit_long = (rsi_1h[i] > 70) or (not trend_bullish)
        exit_short = (rsi_1h[i] < 30) or (not trend_bearish)
        
        # === STOPLOSS LOGIC (ATR-based) ===
        stoploss_distance = 2.0 * atr_1h[i]
        
        if position_side == 1:  # Long position
            # Update highest price for trailing
            highest_price = max(highest_price, close[i])
            
            # Check stoploss (price dropped below entry - 2*ATR)
            if close[i] < entry_price - stoploss_distance:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
            
            # Check trailing stop (price dropped from highest - 2*ATR)
            if close[i] < highest_price - stoploss_distance:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
            
            # Take profit at 2R (reduce to half)
            profit_target = entry_price + 2.0 * stoploss_distance
            if close[i] >= profit_target and signals[i-1] == BASE_SIZE:
                signals[i] = HALF_SIZE
                continue
            
            # Exit on opposite signal
            if exit_long:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
        
        elif position_side == -1:  # Short position
            # Update lowest price for trailing
            lowest_price = min(lowest_price, close[i])
            
            # Check stoploss (price rose above entry + 2*ATR)
            if close[i] > entry_price + stoploss_distance:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
            
            # Check trailing stop (price rose from lowest + 2*ATR)
            if close[i] > lowest_price + stoploss_distance:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
            
            # Take profit at 2R (reduce to half)
            profit_target = entry_price - 2.0 * stoploss_distance
            if close[i] <= profit_target and signals[i-1] == -BASE_SIZE:
                signals[i] = -HALF_SIZE
                continue
            
            # Exit on opposite signal
            if exit_short:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
        
        # === NEW ENTRY ===
        if position_side == 0:
            if long_signal:
                signals[i] = BASE_SIZE
                position_side = 1
                entry_price = close[i]
                highest_price = close[i]
            elif short_signal:
                signals[i] = -BASE_SIZE
                position_side = -1
                entry_price = close[i]
                lowest_price = close[i]
        else:
            # Maintain current position
            signals[i] = signals[i-1]
    
    return signals