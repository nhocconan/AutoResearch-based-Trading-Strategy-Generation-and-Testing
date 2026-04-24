#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA(34) trend filter and 1d volume spike confirmation.
- Primary timeframe: 1h for entries/exits.
- HTF: 4h EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 1h volume > 2.0 * 20-period 1d volume MA to avoid false breakouts.
- Entry: Long when price breaks above Camarilla R3 AND 4h EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla S3 AND 4h EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout (R3/S3) or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Why it should work: Camarilla pivots identify key intraday support/resistance levels.
  In trending markets (4h EMA34 filter), breaks of R3/S3 often continue with momentum.
  Volume spike confirms institutional participation. Session filter avoids Asian session noise.
  Works in bull (long bias) and bear (short bias) via EMA34 trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours filter (08:00-20:00 UTC)
    # prices.index is DatetimeIndex, .hour works directly
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla pivots (based on previous day's OHLC)
    # We need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate typical Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # R2 = C + ((H-L)*1.1/6)
    # R1 = C + ((H-L)*1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L)*1.1/12)
    # S2 = C - ((H-L)*1.1/6)
    # S3 = C - ((H-L)*1.1/4)
    
    # Use previous day's OHLC to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align daily Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 4h data for EMA(34) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 4h close
    ema_34 = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Get 1d data for volume MA (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 1h volume > 2.0 * 20-period 1d volume MA
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 4h bars for EMA34 and 1d bars for volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session filter
            if volume_spike[i]:
                # Bullish breakout: price breaks above Camarilla R3 AND 4h EMA34 bullish (price > EMA34)
                if curr_high > r3 and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below Camarilla S3 AND 4h EMA34 bearish (price < EMA34)
                elif curr_low < s3 and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR loss of volume confirmation OR outside session
            if curr_low < s3 or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR loss of volume confirmation OR outside session
            if curr_high > r3 or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_4hEMA34Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0