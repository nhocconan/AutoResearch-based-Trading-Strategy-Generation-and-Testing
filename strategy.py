#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Primary signal: Break of Camarilla H3/L3 levels from prior 4h bar
# - Trend filter: 4h close > 4h EMA20 for longs, < EMA20 for shorts
# - Volume confirmation: 1h volume > 20-period EMA volume (avoid low-participation signals)
# - Session filter: Trade only 08-20 UTC to reduce noise
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) for 1h strategy
# - Works in bull/bear: Camarilla levels provide adaptive support/resistance,
#   4h EMA20 filter ensures alignment with higher timeframe trend,
#   volume confirmation avoids false breakouts

name = "1h_4h_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA20 for trend direction
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA20 to 1h timeframe (completed 4h bar only)
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h price data
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h volume regime: volume > 20-period EMA volume
    ema_volume_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_regime = volume > ema_volume_20
    
    # Precompute Camarilla levels for each 4h bar
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3_4h = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_l3_4h = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_20_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_regime[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 OR closes below 4h EMA20
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above H3 OR closes above 4h EMA20
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and 4h trend filter
            # Long: price breaks above H3 AND volume regime AND close above 4h EMA20
            if close[i] > camarilla_h3_aligned[i] and volume_regime[i] and close[i] > ema_20_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short: price breaks below L3 AND volume regime AND close below 4h EMA20
            elif close[i] < camarilla_l3_aligned[i] and volume_regime[i] and close[i] < ema_20_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals