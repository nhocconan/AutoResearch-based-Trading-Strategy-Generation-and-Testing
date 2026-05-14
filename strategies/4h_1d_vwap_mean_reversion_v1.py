#!/usr/bin/env python3
# 4h_1d_vwap_mean_reversion_v1
# Strategy: 4-hour VWAP mean reversion with 1-day trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Price reverts to daily VWAP during ranging markets, with trend filter to avoid
# counter-trend trades. Works in both bull and bear by capturing mean reversion within the
# dominant daily trend. Uses volume confirmation to ensure institutional participation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_vwap_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h data for VWAP calculation (to avoid look-ahead)
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # 1d OHLC for trend filter
    close_1d = df_1d['close'].values
    
    # 1h data for VWAP calculation
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price_1h = (high_1h + low_1h + close_1h) / 3.0
    pv_1h = typical_price_1h * volume_1h
    
    # Calculate cumulative VWAP (resets daily)
    cum_pv = np.cumsum(pv_1h)
    cum_volume = np.cumsum(volume_1h)
    vwap_1h = np.where(cum_volume > 0, cum_pv / cum_volume, typical_price_1h)
    
    # Reset VWAP at daily boundaries (00:00 UTC)
    # Assuming 1h data aligns with calendar days
    vwap_1h_daily = vwap_1h.copy()
    # Simple approach: reset every 24 hours (24 * 1h bars)
    for i in range(24, len(vwap_1h), 24):
        if i < len(vwap_1h):
            vwap_1h_daily[i] = typical_price_1h[i]  # Reset at start of day
            # Recalculate cumulative from this point
            cum_pv = np.nancumsum(pv_1h[i:])  # Not ideal but works for demonstration
            # Better: use pandas to group by date
    
    # Alternative: calculate VWAP per day using date grouping
    # For simplicity, we'll use a rolling window that approximates daily VWAP
    # Reset every 24 bars (1 day of 1h data)
    vwap_reset = np.full_like(vwap_1h, np.nan)
    for i in range(0, len(vwap_1h), 24):
        end_idx = min(i + 24, len(vwap_1h))
        if end_idx > i:
            cum_pv_day = np.nansum(pv_1h[i:end_idx])
            cum_vol_day = np.nansum(volume_1h[i:end_idx])
            if cum_vol_day > 0:
                vwap_day = cum_pv_day / cum_vol_day
                vwap_reset[i:end_idx] = vwap_day
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d data to 4h timeframe (wait for daily close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align 1h VWAP to 4h timeframe (1h data is already lower than 4h, so we need to upsample)
    # Since we have 1h data, we can use the last VWAP value within each 4h period
    # Simpler: use the VWAP from the 1h bar that aligns with 4h close
    # We'll use the VWAP value from 1h, 2h, 3h, or 4h ago depending on position in 4h cycle
    # For 4h bar at index i, we use 1h VWAP from the same absolute time
    # Since 1h data is available, we can align it directly
    vwap_1h_aligned = align_htf_to_ltf(prices, df_1h, vwap_reset)
    
    # Standard deviation of price from VWAP for z-score
    price_dev = close - vwap_1h_aligned
    # Use 20-period rolling std of price deviation
    price_dev_ma = pd.Series(price_dev).rolling(window=20, min_periods=20).mean().values
    price_dev_std = pd.Series(price_dev - price_dev_ma).rolling(window=20, min_periods=20).std().values
    # Avoid division by zero
    price_dev_std = np.where(price_dev_std == 0, 1e-10, price_dev_std)
    z_score = (price_dev - price_dev_ma) / price_dev_std
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap_1h_aligned[i]) or 
            np.isnan(z_score[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        z = z_score[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Mean reversion signals: extreme Z-score reversals
        # Long when significantly below VWAP in uptrend (pullback to mean)
        long_signal = (z < -2.0) and vol_spike[i] and uptrend_1d
        # Short when significantly above VWAP in downtrend (pullback to mean)
        short_signal = (z > 2.0) and vol_spike[i] and downtrend_1d
        
        # Exit when price returns to VWAP (Z-score near zero) or opposite extreme
        exit_long = position == 1 and (z > -0.5)  # Return toward mean
        exit_short = position == -1 and (z < 0.5)  # Return toward mean
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals