#!/usr/bin/env python3
"""
1d_FundingRate_MeanReversion_With_Regime
Hypothesis: Funding rate mean reversion works on BTC/ETH in both bull and bear markets.
When weekly funding rate Z-score (30-day) < -2, go long; > +2, go short.
Use 1d timeframe for entries, with 1w funding data as HTF filter.
Add volatility regime filter: only trade when ATR(30) < 1.5 * ATR(90) (low vol regime).
Target: 10-20 trades/year (40-80 total over 4 years) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_90 = pd.Series(tr).ewm(span=90, adjust=False, min_periods=90).mean().values
    vol_regime = atr_30 < (1.5 * atr_90)  # Low volatility regime
    
    # Load weekly funding rate data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate funding rate Z-score (30-week lookback)
    funding = df_1w['funding_rate'].values if 'funding_rate' in df_1w.columns else np.zeros(len(df_1w))
    if len(funding) < 30:
        funding_z = np.zeros(len(funding))
    else:
        funding_mean = pd.Series(funding).rolling(window=30, min_periods=30).mean().values
        funding_std = pd.Series(funding).rolling(window=30, min_periods=30).std().values
        funding_std = np.where(funding_std == 0, 1e-10, funding_std)  # Avoid division by zero
        funding_z = (funding - funding_mean) / funding_std
    
    # Align funding Z-score to 1d timeframe (with 1-week delay for completion)
    funding_z_aligned = align_htf_to_ltf(prices, df_1w, funding_z)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 90  # Warmup for ATR90
    
    for i in range(start_idx, n):
        if (np.isnan(atr_30[i]) or np.isnan(atr_90[i]) or 
            np.isnan(funding_z_aligned[i]) or np.isnan(vol_regime[i])):
            signals[i] = 0.0
            continue
        
        vol_ok = vol_regime[i]
        fz = funding_z_aligned[i]
        
        if position == 0:
            # Long when funding extremely negative (oversold)
            if fz < -2.0 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short when funding extremely positive (overbought)
            elif fz > 2.0 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit when funding returns to neutral or volatility increases
            if fz > -0.5 or not vol_ok:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit when funding returns to neutral or volatility increases
            if fz < 0.5 or not vol_ok:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_FundingRate_MeanReversion_With_Regime"
timeframe = "1d"
leverage = 1.0