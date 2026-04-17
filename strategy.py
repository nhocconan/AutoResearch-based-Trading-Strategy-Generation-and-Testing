#!/usr/bin/env python3
"""
Hypothesis: In the 1-hour timeframe, price respects the 4-hour high/low as key support/resistance levels,
especially during active trading sessions (08-20 UTC). We combine this with a 1-day EMA50 trend filter
and volume confirmation to capture breakouts with controlled frequency.
Long when price breaks above prior 4h high with volume > 1.5x average and price above 1d EMA50.
Short when price breaks below prior 4h low with volume > 1.5x average and price below 1d EMA50.
Exit when price returns to the prior 4h midpoint (mean reversion) or on opposite breakout.
Designed for 1h to work in trending (breakouts) and ranging (mean reversion to mid-point) markets
with ~20-30 trades per year per symbol, avoiding excessive churn.
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
    
    # Get 4h data for prior period's high/low
    df_4h = get_htf_data(prices, '4h')
    
    # Prior 4h high and low (use shift(1) to avoid look-ahead: use completed period's levels)
    phigh = df_4h['high'].shift(1).values
    plow = df_4h['low'].shift(1).values
    pclose = df_4h['close'].values
    
    # Prior 4h midpoint for mean reversion exit
    pmid = (phigh + plow) / 2
    
    # Calculate 1d EMA50 for trend filter (use prior period's close to avoid look-ahead)
    ema_50 = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 4h levels to 1h timeframe (waits for 4h bar to close)
    phigh_1h = align_htf_to_ltf(prices, df_4h, phigh)
    plow_1h = align_htf_to_ltf(prices, df_4h, plow)
    pmid_1h = align_htf_to_ltf(prices, df_4h, pmid)
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: 24-period volume MA on 1h (1 day)
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(phigh_1h[i]) or np.isnan(plow_1h[i]) or np.isnan(pmid_1h[i]) or
            np.isnan(ema_50_1h[i]) or np.isnan(volume_ma_24.iloc[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            # Outside session: flatten or hold flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior 4h high with volume spike and above 1d EMA50
            if price > phigh_1h[i] and vol > 1.5 * vol_ma and price > ema_50_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below prior 4h low with volume spike and below 1d EMA50
            elif price < plow_1h[i] and vol > 1.5 * vol_ma and price < ema_50_1h[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior 4h midpoint (mean reversion) OR breaks below prior 4h low (invalidates breakout)
            if price < pmid_1h[i] or price < plow_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns to prior 4h midpoint (mean reversion) OR breaks above prior 4h high (invalidates breakout)
            if price > pmid_1h[i] or price > phigh_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Prior4HL_Breakout_MeanRev_Session"
timeframe = "1h"
leverage = 1.0