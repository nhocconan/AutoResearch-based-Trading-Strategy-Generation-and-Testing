#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI (VRSI) with 1d ADX regime filter
# VRSI = RSI calculated on volume-weighted price (typical price * volume) / volume
# More sensitive to institutional participation than price-only RSI
# Long when VRSI < 30 and 1d ADX > 25 (trending up)
# Short when VRSI > 70 and 1d ADX > 25 (trending down)
# Volume weighting filters out low-conviction moves, ADX ensures we only trade strong trends
# Works in bull/bear: trends persist across regimes, volume confirms real money participation
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe

name = "6h_VRSI_ADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # Volume-weighted typical price
    vol_weighted_tp = typical_price * volume
    
    # Calculate VRSI (RSI of volume-weighted typical price)
    # RSI calculation: gain/loss over period, then RS = avg_gain/avg_loss, RSI = 100 - (100/(1+RS))
    vol_weighted_series = pd.Series(vol_weighted_tp)
    delta = vol_weighted_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    
    # Avoid division by zero
    rs = avg_gain / (avg_loss + 1e-10)
    v_rsi = 100.0 - (100.0 / (1.0 + rs))
    v_rsi = v_rsi.values
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing)
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1.0/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1.0/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1.0/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100.0 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100.0 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1.0/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, tr_period) + 5  # warmup for VRSI and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(v_rsi[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_vrsi = v_rsi[i]
        curr_adx = adx_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade in strong trends (ADX > 25)
            if curr_adx > 25.0:
                # Long: oversold in uptrend
                if curr_vrsi < 30.0:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought in downtrend
                elif curr_vrsi > 70.0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when VRSI returns to neutral territory (50)
            if curr_vrsi >= 50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when VRSI returns to neutral territory (50)
            if curr_vrsi <= 50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals