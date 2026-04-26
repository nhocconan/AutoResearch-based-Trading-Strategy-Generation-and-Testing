#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_FundingZ_Contrarian_v1
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter, volume spike (2.0x), and funding rate mean reversion (Z-score < -2 long, > +2 short) for BTC/ETH edge. Designed for low overtrading (<30 trades/year) by combining price structure with funding extremes. Works in bull/bear via 1d trend alignment and funding contrarianism.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels from 1d OHLC (using previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: volume > 2.0 * volume_ma(20) for stricter confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Funding rate mean reversion (BTC/ETH edge)
    funding_path = f"data/processed/funding/{'BTCUSDT' if 'BTC' in str(prices.attrs.get('symbol', '')) else 'ETHUSDT' if 'ETH' in str(prices.attrs.get('symbol', '')) else 'BTCUSDT'}.parquet"
    try:
        df_funding = pd.read_parquet(funding_path)
        funding_rate = df_funding['funding_rate'].values
        funding_ma = pd.Series(funding_rate).rolling(window=30, min_periods=30).mean().values
        funding_std = pd.Series(funding_rate).rolling(window=30, min_periods=30).std().values
        funding_z = (funding_rate - funding_ma) / funding_std
        funding_z = np.nan_to_num(funding_z, nan=0.0)
        funding_z_aligned = align_htf_to_ltf(prices, df_funding, funding_z)
        funding_long = funding_z_aligned < -2.0  # Extreme negative = long
        funding_short = funding_z_aligned > 2.0   # Extreme positive = short
    except:
        # Fallback if funding data not available
        funding_long = np.zeros(n, dtype=bool)
        funding_short = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA, 30 for funding)
    start_idx = max(50, 20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with tight filters
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1d uptrend AND volume spike (2.0x) AND funding extreme negative
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and volume_spike[i] and funding_long[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND 1d downtrend AND volume spike (2.0x) AND funding extreme positive
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and volume_spike[i] and funding_short[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 1d trend turns down OR funding normalizes
            if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1 or funding_z_aligned[i] > -0.5:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 1d trend turns up OR funding normalizes
            if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1 or funding_z_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_FundingZ_Contrarian_v1"
timeframe = "4h"
leverage = 1.0