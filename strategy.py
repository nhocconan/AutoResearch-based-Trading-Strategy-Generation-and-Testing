#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Primary signal: 1h price breaks above/below Camarilla H3/L3 levels from prior 4h candle
# - Trend filter: 4h EMA50 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 1h volume > 20-period median volume (avoid low-participation signals)
# - Session filter: Trade only 08-20 UTC to reduce noise
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Camarilla pivots provide structure in ranging markets, EMA50 filter
#   ensures trades align with higher timeframe trend, reducing false signals in strong trends

name = "1h_4h_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (completed 4h bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    # Calculate Camarilla levels for each 1h bar using prior 4h bar's OHLC
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    # For each 1h bar, use the most recent completed 4h bar's OHLC
    for i in range(n):
        # Find index of completed 4h bar (4h bar that closed at or before this 1h bar)
        # Each 4h bar = 4 * 1h bars, so 4h bar index = i // 4
        h4_idx = i // 4
        if h4_idx < len(df_4h):
            # Use prior completed 4h bar (h4_idx - 1) to avoid look-ahead
            if h4_idx >= 1:
                h4_high = high_4h[h4_idx - 1]
                h4_low = low_4h[h4_idx - 1]
                h4_close = close_4h[h4_idx - 1]
                camarilla_h3[i] = h4_close + (h4_high - h4_low) * 1.1 / 4
                camarilla_l3[i] = h4_close - (h4_high - h4_low) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(camarilla_h3[i]) or
            np.isnan(camarilla_l3[i]) or
            np.isnan(volume_regime[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below camarilla l3 OR price crosses below 4h EMA50
            if close[i] < camarilla_l3[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above camarilla h3 OR price crosses above 4h EMA50
            if close[i] > camarilla_h3[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 4h EMA50 filter
            # Long: price breaks above camarilla h3 AND volume regime AND price above 4h EMA50
            if close[i] > camarilla_h3[i] and volume_regime[i] and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short: price breaks below camarilla l3 AND volume regime AND price below 4h EMA50
            elif close[i] < camarilla_l3[i] and volume_regime[i] and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals