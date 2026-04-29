#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter + volume spike confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# In bull/bear markets: Alligator alignment (Lips > Teeth > Jaw for uptrend, inverse for downtrend)
# combined with 1d EMA34 trend filter and volume spike provides high-probability entries.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Uses discrete position sizing (0.25) to reduce churn.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: SMAs of median price (typical price) with different periods
    typical_price = (high + low + close) / 3.0
    
    # Alligator components: Jaw (13-period, 8 bars ahead), Teeth (8-period, 5 bars ahead), Lips (5-period, 3 bars ahead)
    # We calculate the SMAs and then shift them forward to avoid look-ahead
    ma_tp = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().values  # Lips base
    ma_teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values  # Teeth base
    ma_jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().values  # Jaw base
    
    # Shift forward to represent Alligator's forward projection (no look-ahead)
    # Lips: 5-period SMA shifted 3 bars forward
    lips = np.concatenate([np.full(3, np.nan), ma_tp[:-3]]) if len(ma_tp) > 3 else np.full(n, np.nan)
    # Teeth: 8-period SMA shifted 5 bars forward
    teeth = np.concatenate([np.full(5, np.nan), ma_teeth[:-5]]) if len(ma_teeth) > 5 else np.full(n, np.nan)
    # Jaw: 13-period SMA shifted 8 bars forward
    jaw = np.concatenate([np.full(8, np.nan), ma_jaw[:-8]]) if len(ma_jaw) > 8 else np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 34, 20, 13)  # warmup for EMA34, Alligator, volume
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator alignment breaks (Lips < Teeth or Teeth < Jaw) - trend weakening
            # 2. Price crosses below 1d EMA34 (HTF trend change)
            if (curr_lips < curr_teeth or curr_teeth < curr_jaw or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator alignment breaks (Lips > Teeth or Teeth > Jaw) - trend weakening
            # 2. Price crosses above 1d EMA34 (HTF trend change)
            if (curr_lips > curr_teeth or curr_teeth > curr_jaw or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Alligator aligned for uptrend (Lips > Teeth > Jaw) + above 1d EMA34 + volume confirm
            if (curr_lips > curr_teeth and curr_teeth > curr_jaw and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Alligator aligned for downtrend (Lips < Teeth < Jaw) + below 1d EMA34 + volume confirm
            elif (curr_lips < curr_teeth and curr_teeth < curr_jaw and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals