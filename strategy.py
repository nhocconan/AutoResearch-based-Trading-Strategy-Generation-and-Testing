#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout + 4h EMA34 trend filter + volume spike confirmation + session filter (08-20 UTC)
- Uses 1h Camarilla pivot levels H3/L3 as intraday breakout levels
- 4h EMA34 as HTF trend filter to ensure alignment with higher timeframe momentum
- Volume spike (2.0x 20-period MA) confirms institutional participation and reduces false breakouts
- Session filter (08-20 UTC) avoids low-liquidity Asian session noise
- Discrete position sizing (0.20) minimizes fee churn
- Target: 15-37 trades/year per symbol (~60-150 total over 4 years)
- Works in bull markets (buying H3 breakouts in uptrend) and bear markets (selling L3 breakouts in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for filter
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1h data for Camarilla calculation (primary timeframe)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Get 4h data for EMA34 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels on 1h (using previous bar's OHLC)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1h + 1.1 * (high_1h - low_1h) / 2
    camarilla_l3 = close_1h - 1.1 * (high_1h - low_1h) / 2
    
    # Calculate EMA34 on 4h for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 1h
    volume_ma_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1h timeframe (primary)
    h3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_l3)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        ema_trend = ema34_4h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        hour = hours[i]
        
        # Session filter: 08-20 UTC
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above H3 + volume spike + price > 4h EMA34 (uptrend)
            if price > h3 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3 + volume spike + price < 4h EMA34 (downtrend)
            elif price < l3 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below L3 (mean reversion) or trend breaks
            if price < l3 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above H3 (mean reversion) or trend breaks
            if price > h3 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0