#!/usr/bin/env python3
"""
exp_7395_6d_vix_cross_signal_v1
Hypothesis: 6-day VIX-like volatility index crossing above/below its 34-period EMA with volume confirmation.
Uses high-low range as volatility proxy. Works in both bull/bear by capturing volatility expansion/contraction cycles.
Targets 80-180 trades over 4 years (20-45/year). Discrete sizing (0.0, ±0.25) minimizes fee churn.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7395_6d_vix_cross_signal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOL_LOOKBACK = 14          # Period for volatility calculation (similar to VIX)
VIX_EMA_FAST = 34          # Fast EMA for volatility index
VIX_EMA_SLOW = 89          # Slow EWA for volatility trend filter
VOL_SPIKE_MULT = 2.0       # Volume must be 2x average to confirm
SIGNAL_SIZE = 0.25         # Position size as fraction of capital
ATR_PERIOD = 14            # ATR for stop loss
ATR_STOP_MULT = 2.5        # Stop loss distance
MAX_HOLD_BARS = 12         # Maximum hold time (3 days)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate volatility index (high-low range normalized)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True range components
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    # Handle first element
    tr1[0] = high[0] - low[0]
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    
    # Volatility index = smoothed true range (similar to VIX calculation)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    vol_index = pd.Series(tr).ewm(span=VOL_LOOKBACK, adjust=False).values
    
    # VIX EMAs for signal generation
    vix_ema_fast = pd.Series(vol_index).ewm(span=VIX_EMA_FAST, adjust=False).values
    vix_ema_slow = pd.Series(vol_index).ewm(span=VIX_EMA_SLOW, adjust=False).values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False).values
    
    # ATR for stop loss
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start after warmup period
    start = max(VIX_EMA_SLOW, 20) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if data not ready
        if np.isnan(vix_ema_fast[i]) or np.isnan(vix_ema_slow[i]) or np.isnan(atr[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stop loss
        if position == 1:  # Long position
            if close[i] <= entry_price - ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # Short position
            if close[i] >= entry_price + ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_SPIKE_MULT if not np.isnan(vol_ma[i]) else False
        
        # VIX crossover signals
        vix_bullish = vix_ema_fast[i] > vix_ema_slow[i]
        vix_bearish = vix_ema_fast[i] < vix_ema_slow[i]
        
        # Volatility expansion/contraction signals
        vix_expanding = vix_ema_fast[i] > vix_ema_fast[i-1] and vix_ema_slow[i] > vix_ema_slow[i-1]
        vix_contracting = vix_ema_fast[i] < vix_ema_fast[i-1] and vix_ema_slow[i] < vix_ema_slow[i-1]
        
        # Entry logic: 
        # Long: VIX crosses above slow EMA during volatility expansion (fear increasing -> potential reversal)
        # Short: VIX crosses below slow EMA during volatility contraction (calm before storm)
        if position == 0:
            # Long when VIX fast crosses above slow AND volatility is expanding
            if vix_ema_fast[i] > vix_ema_slow[i] and vix_ema_fast[i-1] <= vix_ema_slow[i-1] and vix_expanding and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Short when VIX fast crosses below slow AND volatility is contracting
            elif vix_ema_fast[i] < vix_ema_slow[i] and vix_ema_fast[i-1] >= vix_ema_slow[i-1] and vix_contracting and vol_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold position
            signals[i] = position * SIGNAL_SIZE
    
    return signals