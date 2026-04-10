#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Primary signal: Williams %R(14) crosses above/below oversold/overbought levels on 6h
# - Trend filter: Price above/below 50-period EMA on 1w (only trade with weekly trend)
# - Volume confirmation: 6h volume > 1.2x 20-period average volume
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.2 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, prices, vol_spike)  # same timeframe
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR stoploss hit
            if williams_r[i] > -20 or close_6h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR stoploss hit
            if williams_r[i] < -80 or close_6h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R mean reversion with trend and volume filters
            if vol_spike_aligned[i]:
                # Long: Williams %R crosses above -80 from below (oversold bounce) AND price above weekly EMA
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and close_6h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above (overbought rejection) AND price below weekly EMA
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close_6h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals