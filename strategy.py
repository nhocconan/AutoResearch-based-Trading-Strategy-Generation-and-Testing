#!/usr/bin/env python3
"""
Experiment #139: 4h Primary + 1d HTF — Fisher Transform Reversals + Volume Confirmation

Hypothesis: After 121 failed experiments, the pattern shows:
- Complex regime filters (Choppiness, ADX dual-regime) cause 0 trades
- RSI-only strategies work but get whipsawed in 2025 bear market
- Fisher Transform catches reversals better than RSI in bear/range markets (research shows 75% win rate)
- Volume confirmation reduces false signals without killing trade frequency
- 4h + 1d combination has proven success (SOL +0.879, ETH +0.755 in history)

This strategy uses:
1. 1d HMA(21) = major trend bias (price above/below)
2. 4h Fisher Transform(9) = entry trigger (crosses -1.5 long, +1.5 short)
3. Volume > 1.2 * SMA_vol(20) = confirmation (avoids low-liquidity traps)
4. ATR(14) trailing stoploss 2.5x for risk management
5. Position size 0.27 (27% of capital, conservative)

Key differences from failed experiments:
- Fisher Transform instead of RSI (better for bear market reversals)
- Volume confirmation (missing in most failed 4h strategies)
- Simpler HTF bias (HMA vs KAMA - less lag)
- Loose Fisher thresholds (-1.5/+1.5) to ensure trade generation

Target: Sharpe>0.351, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_vol_hma_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper: Weighted Moving Average
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    close_series = pd.Series(close)
    
    # WMA(n/2)
    wma_half = wma(close, period // 2)
    # WMA(n)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n) period
    sqrt_period = int(np.sqrt(period))
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_fisher(close, high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = EMA of normalized price
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Normalize price to -1 to +1 range using highest high / lowest low
    hh = np.zeros(n)
    ll = np.zeros(n)
    
    hh[0] = high[0]
    ll[0] = low[0]
    
    for i in range(1, n):
        hh[i] = max(hh[i-1], high[i])
        ll[i] = min(ll[i-1], low[i])
        
        # Reset every period to avoid stale extremes
        if i >= period:
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
    
    # Normalize: (2 * (close - LL) / (HH - LL)) - 1
    x = np.zeros(n)
    for i in range(period, n):
        range_val = hh[i] - ll[i]
        if range_val > 1e-10:
            x[i] = 2.0 * (close[i] - ll[i]) / range_val - 1.0
        else:
            x[i] = 0.0
    
    # Clamp to avoid division by zero in Fisher formula
    x = np.clip(x, -0.999, 0.999)
    
    # EMA of x
    x_ema = pd.Series(x).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Fisher Transform
    for i in range(period, n):
        if abs(x_ema[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + x_ema[i]) / (1.0 - x_ema[i]))
        else:
            fisher[i] = fisher[i-1] if i > period else 0.0
    
    return fisher

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher = calculate_fisher(close, high, low, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.27  # 27% position size (conservative for 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crosses
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (prev_fisher < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_short = (prev_fisher > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow entries when Fisher is at extremes (not just crosses)
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + (Fisher cross long OR Fisher extreme long) + volume confirmed
        # SHORT: 1d bear + (Fisher cross short OR Fisher extreme short) + volume confirmed
        desired_signal = 0.0
        
        if htf_bull and (fisher_cross_long or fisher_extreme_long) and vol_confirmed:
            desired_signal = SIZE
        elif htf_bear and (fisher_cross_short or fisher_extreme_short) and vol_confirmed:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
        prev_fisher = fisher[i]
    
    return signals