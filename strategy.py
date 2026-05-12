#!/usr/bin/env python3
name = "1d_FundingRateMeanReversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_time = prices['open_time'].values
    
    # Load funding rate data for the symbol
    symbol = getattr(prices, 'symbol', 'BTCUSDT').replace('USDT', '')
    funding_path = f"data/processed/funding/{symbol}_funding_rate_8h.parquet"
    
    try:
        funding_df = pd.read_parquet(funding_path)
        # Ensure datetime index for merge
        funding_df = funding_df.set_index('funding_time')
        # Align funding rate to price data using forward fill (funding known at 8h intervals)
        funding_series = pd.Series(index=open_time, dtype=float)
        for i, dt in enumerate(open_time):
            # Find the most recent funding rate before or at this time
            mask = funding_df.index <= dt
            if mask.any():
                funding_series.iloc[i] = funding_df.loc[mask, 'funding_rate'].iloc[-1]
            else:
                funding_series.iloc[i] = np.nan
        funding_rate = funding_series.values
    except:
        # If funding data not available, return no signals
        return np.zeros(n)
    
    # Calculate 30-day z-score of funding rate
    funding_rate_series = pd.Series(funding_rate)
    funding_mean = funding_rate_series.rolling(window=360, min_periods=360).mean()  # 30 days * 8h = 240, but use 360 for more stability
    funding_std = funding_rate_series.rolling(window=360, min_periods=360).std()
    funding_z = (funding_rate - funding_mean) / funding_std
    
    # Load weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA 34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 360  # Need enough data for z-score calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(funding_z[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: funding rate extremely negative (mean reversion long) + weekly uptrend
            if (funding_z[i] < -2.0 and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: funding rate extremely positive (mean reversion short) + weekly downtrend
            elif (funding_z[i] > 2.0 and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: funding rate normalizes or trend breaks
            if funding_z[i] > -0.5 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: funding rate normalizes or trend breaks
            if funding_z[i] < 0.5 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals