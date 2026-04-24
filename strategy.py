#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA(34) trend filter and 1d volume spike confirmation.
- Primary timeframe: 1h for entries/exits.
- HTF: 4h EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 1h volume > 2.0 * 20-period 1d volume MA to avoid false breakouts.
- Entry: Long when price breaks above Camarilla H3 level AND 4h EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla L3 level AND 4h EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in both bull and bear markets: EMA34 trend filter adapts to market regime,
  volume spike confirms institutional participation, Camarilla levels provide
  precise intraday support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    # prices.index is already DatetimeIndex with no conversion needed
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels (H3, L3) from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Camarilla width = (high - low) * 1.1 / 12
    camarilla_width = (high - low) * 1.1 / 12.0
    # H3 = close + width * 1.1
    camarilla_h3 = close + camarilla_width * 1.1
    # L3 = close - width * 1.1
    camarilla_l3 = close - camarilla_width * 1.1
    
    # Get 4h data for EMA(34) trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 4h close
    ema_34 = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 1h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 4h bars for EMA34 and 1d bars for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_camarilla = camarilla_h3[i]
        lower_camarilla = camarilla_l3[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session filter
            if volume_spike[i]:
                # Bullish breakout: price breaks above Camarilla H3 AND 4h EMA34 bullish (price > EMA34)
                if curr_high > upper_camarilla and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below Camarilla L3 AND 4h EMA34 bearish (price < EMA34)
                elif curr_low < lower_camarilla and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla L3 OR loss of volume confirmation OR outside session
            if curr_low < lower_camarilla or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above Camarilla H3 OR loss of volume confirmation OR outside session
            if curr_high > upper_camarilla or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34Trend_1dVolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0