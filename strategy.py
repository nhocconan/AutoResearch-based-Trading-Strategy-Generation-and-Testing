#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h Trend Filter and Session Filter
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance levels
# 4h EMA50 filter ensures we trade breakouts in direction of higher timeframe trend
# Session filter (08-20 UTC) avoids low-liquidity Asian session noise
# Works in bull markets (buy breakouts above R3 in uptrend) and bear markets (sell breakdowns below S3 in downtrend)
# Target: 15-37 trades/year (60-150 total over 4 years)

name = "1h_Camarilla_R3S3_Breakout_4hTrend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 4h calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla pivot points (R3, S3) from previous day
    # Typical Camarilla formula based on previous day's range
    # We'll approximate using rolling 24-period (1 day) high/low/close for 1h data
    lookback = 24  # 24 hours = 1 day
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    roll_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = roll_close + (roll_high - roll_low) * 1.1 / 4
    camarilla_s3 = roll_close - (roll_high - roll_low) * 1.1 / 4
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma_24)
    
    # Session filter: 08-20 UTC (avoid Asian session low liquidity)
    hours = prices.index.hour  # pre-computed DatetimeIndex hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, lookback, 24)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_4h = ema50_4h_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_volume_confirm = volume_confirm[i]
        curr_hour = hours[i]
        
        # Session filter: only trade 08-20 UTC
        in_session = (8 <= curr_hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Look for breakout at Camarilla R3/S3 levels with volume confirmation
            # Bullish breakout: price breaks above R3 in bullish 4h trend
            if curr_high > curr_r3 and curr_volume_confirm and curr_close > curr_ema50_4h:
                signals[i] = 0.20
                position = 1
            # Bearish breakout: price breaks below S3 in bearish 4h trend
            elif curr_low < curr_s3 and curr_volume_confirm and curr_close < curr_ema50_4h:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to Camarilla H4 level (mean reversion) or trend breaks
            camarilla_h4 = roll_close[i] + (roll_high[i] - roll_low[i]) * 1.1 / 2  # H4 level
            if curr_close < camarilla_h4 or curr_close < curr_ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to Camarilla L4 level (mean reversion) or trend breaks
            camarilla_l4 = roll_close[i] - (roll_high[i] - roll_low[i]) * 1.1 / 2  # L4 level
            if curr_close > camarilla_l4 or curr_close > curr_ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals