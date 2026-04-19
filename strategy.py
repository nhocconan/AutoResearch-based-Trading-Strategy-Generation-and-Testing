#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Wilder's Volatility VIX Fix with 1-day ATR and volume confirmation.
# Long when: VIX Fix > 90 (extreme fear), price closes above 10-period EMA, volume > 1.8x 20-period average
# Short when: VIX Fix > 90 (extreme fear), price closes below 10-period EMA, volume > 1.8x 20-period average
# Exit when VIX Fix < 50 (reduced fear) or price crosses 10-period EMA in opposite direction.
# Designed for ~20-30 trades/year per symbol. Works in both bull and bear markets by capturing mean reversion during extreme volatility spikes.
name = "4h_WilderVIXFix_Volume_EMA10"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on daily data (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) 
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    
    # Align ATR to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate Wilder's Volatility VIX Fix on 4h data
    # VIX Fix = (Highest High in 22-period - Close) / ATR * 100
    highest_high_22 = pd.Series(high).rolling(window=22, min_periods=22).max().values
    vix_fix = ((highest_high_22 - close) / atr_aligned) * 100
    
    # 10-period EMA for trend filter
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vix_fix[i]) or np.isnan(ema_10[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vix_val = vix_fix[i]
        ema_val = ema_10[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Extreme fear condition: VIX Fix > 90
            if vix_val > 90 and vol > 1.8 * vol_ma:
                if price > ema_val:
                    # Long: extreme fear + price above EMA
                    signals[i] = 0.25
                    position = 1
                elif price < ema_val:
                    # Short: extreme fear + price below EMA
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: fear subsided or price crossed below EMA
            if vix_val < 50 or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: fear subsided or price crossed above EMA
            if vix_val < 50 or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals