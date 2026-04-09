#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Primary signal: Price breaks above/below Camarilla pivot levels (H3/L3) calculated from prior 1d candle
# - Trend filter: 1w EMA50 - ensures alignment with weekly trend (bullish for longs, bearish for shorts)
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation false breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Camarilla levels act as dynamic support/resistance; weekly EMA filter avoids counter-trend trades
# - Exit: Price reverts to Camarilla pivot point (H4/L4) or weekly EMA50 crossover

name = "1d_1w_camarilla_pivot_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d volume regime: volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Need at least 2 days of data to calculate Camarilla levels (yesterday's OHLC)
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Get prior day's OHLC for Camarilla calculation
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        prev_close = prices['close'].iloc[i-1]
        
        # Skip if any required data is invalid
        if (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels from prior day's OHLC
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        pivot = (prev_high + prev_low + prev_close) / 3
        h3 = pivot + (range_val * 1.1 / 4)  # ~1.1 * range / 4
        l3 = pivot - (range_val * 1.1 / 4)  # ~1.1 * range / 4
        h4 = pivot + (range_val * 1.1 / 2)  # ~1.1 * range / 2
        l4 = pivot - (range_val * 1.1 / 2)  # ~1.1 * range / 2
        
        # Current price
        close_price = prices['close'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below H4 (pivot point) OR weekly EMA50 turns bearish
            if close_price < pivot or close_price < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above L4 (pivot point) OR weekly EMA50 turns bullish
            if close_price > pivot or close_price > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and weekly EMA50 filter
            # Long: Price breaks above H3 with volume AND price above weekly EMA50 (bullish bias)
            if (close_price > h3 and 
                volume_regime[i] and 
                close_price > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Price breaks below L3 with volume AND price below weekly EMA50 (bearish bias)
            elif (close_price < l3 and 
                  volume_regime[i] and 
                  close_price < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals