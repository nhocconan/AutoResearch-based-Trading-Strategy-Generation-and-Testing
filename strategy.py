#!/usr/bin/env python3
"""
EXPERIMENT #037 - Volume-Weighted Momentum with Dual HTF Trend Filter (15m primary)
====================================================================================
Hypothesis: 15m entries with dual HTF confirmation (1h + 4h HMA) reduce false signals.
Volume spikes confirm genuine momentum moves. VWAP-based momentum filter ensures
we enter only when price is above/below fair value. This differs from previous
attempts by using volume-weighted signals and requiring BOTH HTF trends to agree.

Key features:
- Primary TF: 15m (this experiment)
- HTF filters: 1h HMA(21) + 4h HMA(21) - BOTH must agree on trend direction
- Momentum: Price vs VWAP(20) deviation (Z-score style)
- Volume confirmation: Volume > 1.5x 20-period average
- Entry: Momentum + Volume + Dual HTF trend alignment
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25 discrete levels (conservative for 15m)
- Take profit: Reduce to half at 2R, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "vwmomentum_dual_htf_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_vwap(high, low, close, volume, period=20):
    """Calculate Volume-Weighted Average Price (rolling)"""
    n = len(close)
    vwap = np.zeros(n)
    vwap[:] = np.nan
    
    for i in range(period - 1, n):
        typical_price = (high[i - period + 1:i + 1] + low[i - period + 1:i + 1] + close[i - period + 1:i + 1]) / 3
        vol = volume[i - period + 1:i + 1]
        vwap[i] = np.sum(typical_price * vol) / (np.sum(vol) + 1e-10)
    
    return vwap


def calculate_momentum_zscore(close, vwap, period=20):
    """Calculate Z-score of price deviation from VWAP"""
    n = len(close)
    deviation = close - vwap
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    for i in range(period - 1, n):
        if not np.isnan(vwap[i]) and vwap[i] > 0:
            window_dev = deviation[i - period + 1:i + 1]
            window_dev = window_dev[~np.isnan(window_dev)]
            if len(window_dev) > 0:
                mean_dev = np.mean(window_dev)
                std_dev = np.std(window_dev) + 1e-10
                zscore[i] = (deviation[i] - mean_dev) / std_dev
    
    return zscore


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    volume_ratio = volume / (vol_avg + 1e-10)
    return volume_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF HMAs
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF indicators to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    vwap = calculate_vwap(high, low, close, volume, 20)
    momentum_z = calculate_momentum_zscore(close, vwap, 20)
    volume_ratio = calculate_volume_ratio(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative for 15m)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vwap[i]) or np.isnan(momentum_z[i]) or 
            np.isnan(volume_ratio[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend filter - BOTH must agree
        hma_1h_trend = 1 if close[i] > hma_1h_aligned[i] else -1
        hma_4h_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Only trade when both HTF trends agree
        dual_htf_aligned = (hma_1h_trend == hma_4h_trend)
        htf_direction = hma_1h_trend if dual_htf_aligned else 0
        
        # Momentum filter: Z-score > 0.5 for long, < -0.5 for short
        momentum_long = momentum_z[i] > 0.5
        momentum_short = momentum_z[i] < -0.5
        
        # Volume confirmation: volume > 1.3x average
        volume_confirmed = volume_ratio[i] > 1.3
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Dual HTF bullish + Momentum positive + Volume confirmed
        if htf_direction == 1 and momentum_long and volume_confirmed:
            target_signal = SIZE
        
        # Short entry: Dual HTF bearish + Momentum negative + Volume confirmed
        elif htf_direction == -1 and momentum_short and volume_confirmed:
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.0*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if HTF trend reversed)
                if position_side == 1 and htf_direction == -1:
                    # HTF trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and htf_direction == 1:
                    # HTF trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals