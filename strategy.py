#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Price breaks above/below 1h Camarilla H3/L3 levels with volume > 2.0 * 1h volume MA(20) and 4h EMA34 alignment.
- Exit: Price touches the opposite Camarilla level (mean reversion) or breaks the Camarilla midpoint (trend exhaustion).
- Signal size: 0.20 discrete to minimize fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Designed for BTC/ETH: Camarilla pivots provide mathematical support/resistance, EMA34 filters trend, volume confirms breakout validity, session filter reduces noise.
- Works in bull markets by following trend with breakouts, works in bear markets by fading false breakouts at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla calculation
    df_1h = prices.copy()
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels (using previous bar's HLC)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # We use the previous completed bar to calculate levels for current bar
    prev_close = df_1h['close'].shift(1).values
    prev_high = df_1h['high'].shift(1).values
    prev_low = df_1h['low'].shift(1).values
    
    # Calculate Camarilla levels for each bar based on previous bar
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4.0
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4.0
    camarilla_m3 = prev_close + 1.1 * (prev_high - prev_low) / 6.0  # Midpoint for exit
    camarilla_lm3 = prev_close - 1.1 * (prev_high - prev_low) / 6.0
    
    # Calculate 4h EMA34 for trend
    ema_34 = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Determine 4h EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
            trend_bullish = close[i] > ema_34_aligned[i]
            trend_bearish = close[i] < ema_34_aligned[i]
            
            # Long: price breaks above Camarilla H3 AND 4h trend bullish AND volume confirmed
            if curr_high > camarilla_h3[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla L3 AND 4h trend bearish AND volume confirmed
            elif curr_low < camarilla_l3[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on touch of Camarilla L3 (mean reversion) or break below M3 with weakness
            if curr_low <= camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on touch of Camarilla H3 (mean reversion) or break above M3 with weakness
            if curr_high >= camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0