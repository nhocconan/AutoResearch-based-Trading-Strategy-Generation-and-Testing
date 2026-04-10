#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI (VW-RSI) + 1d Supertrend filter + ATR-based position sizing
# - VW-RSI: RSI calculation using volume-weighted price instead of close price
# - Long: VW-RSI < 30 (oversold) AND 1d Supertrend = uptrend (filter for trend alignment)
# - Short: VW-RSI > 70 (overbought) AND 1d Supertrend = downtrend
# - Position size scaled by ATR volatility (inverse volatility weighting)
# - Designed for 6h timeframe: targets 50-150 trades over 4 years (12-37/year)
# - Works in bull/bear markets: Supertrend filter ensures we trade with higher timeframe trend
# - Volume weighting reduces noise and improves signal quality during low-volume periods

name = "6h_1d_vwrsi_supertrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Supertrend for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_multiplier = 3.0
    upper_band = (high_1d + low_1d) / 2 + atr_multiplier * atr_10
    lower_band = (high_1d + low_1d) / 2 - atr_multiplier * atr_10
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1d[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = upper_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = lower_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
    
    # Supertrend is uptrend when price > supertrend value
    supertrend_uptrend = close_1d > supertrend
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend_uptrend.astype(float))
    
    # Pre-compute 6h Volume-Weighted RSI (VW-RSI)
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # Typical price (VWAP-style)
    typical_price = (high_6h + low_6h + close_6h) / 3.0
    # Volume-weighted typical price
    vw_price = typical_price * volume_6h
    
    # Calculate price changes
    delta = np.diff(vw_price, prepend=vw_price[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[period] = np.mean(gain[:period+1])
    avg_loss[period] = np.mean(loss[:period+1])
    
    for i in range(period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    vwrsi = 100 - (100 / (1 + rs))
    # For first period values, use simple calculation
    for i in range(period):
        if avg_loss[i] != 0:
            vwrsi[i] = 100 - (100 / (1 + (avg_gain[i] / avg_loss[i])))
        else:
            vwrsi[i] = 100 if avg_gain[i] > 0 else 0
    
    # Pre-compute 6h ATR(14) for volatility-based position sizing
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14 = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Base ATR for normalization (using median ATR)
    base_atr = np.nanmedian(atr_14[~np.isnan(atr_14)]) if np.any(~np.isnan(atr_14)) else 1.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_aligned[i]) or np.isnan(vwrsi[i]) or 
            np.isnan(atr_14[i]) or atr_14[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: VW-RSI > 50 (exit overbought) OR Supertrend turns down
            if vwrsi[i] > 50 or supertrend_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by inverse volatility (ATR)
                vol_factor = base_atr / atr_14[i]
                vol_factor = np.clip(vol_factor, 0.5, 2.0)  # Limit leverage effect
                base_size = 0.25
                signals[i] = base_size * vol_factor
                # Ensure we don't exceed max position size
                signals[i] = np.clip(signals[i], -0.40, 0.40)
                
        elif position == -1:  # Short position
            # Exit: VW-RSI < 50 (exit oversold) OR Supertrend turns up
            if vwrsi[i] < 50 or supertrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by inverse volatility (ATR)
                vol_factor = base_atr / atr_14[i]
                vol_factor = np.clip(vol_factor, 0.5, 2.0)  # Limit leverage effect
                base_size = 0.25
                signals[i] = -base_size * vol_factor
                # Ensure we don't exceed max position size
                signals[i] = np.clip(signals[i], -0.40, 0.40)
        else:  # Flat
            # Look for VW-RSI signals with Supertrend filter
            # Long: VW-RSI < 30 (oversold) AND 1d Supertrend = uptrend
            if vwrsi[i] < 30 and supertrend_aligned[i] > 0.5:
                position = 1
                entry_price = close_6h[i]
                # Scale position by inverse volatility (ATR)
                vol_factor = base_atr / atr_14[i]
                vol_factor = np.clip(vol_factor, 0.5, 2.0)  # Limit leverage effect
                base_size = 0.25
                signals[i] = base_size * vol_factor
                signals[i] = np.clip(signals[i], -0.40, 0.40)
            # Short: VW-RSI > 70 (overbought) AND 1d Supertrend = downtrend
            elif vwrsi[i] > 70 and supertrend_aligned[i] < 0.5:
                position = -1
                entry_price = close_6h[i]
                # Scale position by inverse volatility (ATR)
                vol_factor = base_atr / atr_14[i]
                vol_factor = np.clip(vol_factor, 0.5, 2.0)  # Limit leverage effect
                base_size = 0.25
                signals[i] = -base_size * vol_factor
                signals[i] = np.clip(signals[i], -0.40, 0.40)
    
    return signals