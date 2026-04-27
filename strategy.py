#!/usr/bin/env python3
"""
1d_Weekly_CCI_Mean_Reversion_v1
Hypothesis: Uses weekly CCI to detect overbought/oversold conditions for mean reversion on daily timeframe.
In bull markets, oversold bounces capture dips; in bear markets, overbought reversals catch rallies.
Combined with volume confirmation to avoid false signals and monthly volatility filter to adapt to market regimes.
Target: 15-25 trades per year to minimize fee drag while capturing meaningful reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly CCI (20-period)
    typical_price_weekly = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3
    sma_tp = typical_price_weekly.rolling(window=20, min_periods=20).mean()
    mad = typical_price_weekly.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_weekly = (typical_price_weekly - sma_tp) / (0.015 * mad)
    cci_weekly = cci_weekly.values
    
    # Align weekly CCI to daily with proper delay for indicator confirmation
    cci_aligned = align_htf_to_ltf(prices, df_weekly, cci_weekly)
    
    # Daily volatility filter: monthly (20-day) volatility percentile
    returns = np.diff(np.log(close), prepend=0)
    vol_20d = pd.Series(returns).rolling(window=20, min_periods=20).std() * np.sqrt(252)
    vol_percentile = pd.Series(vol_20d).rolling(window=252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Volume confirmation: current volume > 1.3 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 60  # Need weekly data + monthly volatility
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cci_aligned[i]) or np.isnan(vol_percentile[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        cci_val = cci_aligned[i]
        vol_regime = vol_percentile[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Enter long: weekly oversold (< -100) + low volatility regime (mean reversion works better in low vol) + volume confirmation
            if cci_val < -100 and vol_regime < 0.7 and vol_conf:
                signals[i] = size
                position = 1
            # Enter short: weekly overbought (> 100) + low volatility regime + volume confirmation
            elif cci_val > 100 and vol_regime < 0.7 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: weekly CCI returns to neutral (> -50) or volatility spikes (breakdown risk)
            if cci_val > -50 or vol_regime > 0.9:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: weekly CCI returns to neutral (< 50) or volatility spikes (breakout risk)
            if cci_val < 50 or vol_regime > 0.9:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_CCI_Mean_Reversion_v1"
timeframe = "1d"
leverage = 1.0