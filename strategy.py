#!/usr/bin/env python3
"""
1d_FundingRate_ZScore_MeanReversion_1wTrendFilter
Hypothesis: Funding rate mean reversion with 1-week trend filter works on BTC/ETH in both bull and bear markets.
Extreme negative funding (Z < -2) = long signal in uptrend (1w EMA50 up). Extreme positive funding (Z > +2) = short signal in downtrend (1w EMA50 down).
Uses discrete sizing 0.25 to limit trades (~10-25/year) and avoid fee drag. Volatility filter (ATR ratio > 0.8) ensures entries during sufficient movement.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load funding rate data (assuming available via mtf_data or pre-loaded)
    # For this strategy, we simulate funding rate as a proxy using price momentum
    # In practice, replace with actual funding rate from data/processed/funding/
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d returns for funding rate proxy (actual funding should be loaded externally)
    # Using 8-hour returns as funding proxy: log(close / close.shift(32)) for 1d data (32x 30m bars)
    # But since we are on 1d timeframe, we use daily returns as proxy for funding rate
    # Actual implementation should load funding rate parquet files
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    funding_proxy = pd.Series(returns).rolling(window=8, min_periods=8).mean().values  # 8-day smooth as proxy
    
    # Calculate Z-score of funding rate (proxy) over 30 days
    funding_mean = pd.Series(funding_proxy).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding_proxy).rolling(window=30, min_periods=30).std().values
    funding_z = (funding_proxy - funding_mean) / (funding_std + 1e-9)
    
    # Load 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for volatility filter
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volatility regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for ATR ratio, 30 for funding Z-score, and 50 for EMA alignment
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(funding_z[i]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        funding_z_val = funding_z[i]
        vol_filter = atr_ratio[i] > 0.8  # volatility filter: ensure sufficient movement
        size = fixed_size
        
        # Entry conditions: Extreme funding rate Z-score with 1w trend alignment
        # Long: extremely negative funding (Z < -2) AND 1w uptrend (price > EMA50)
        # Short: extremely positive funding (Z > +2) AND 1w downtrend (price < EMA50)
        long_entry = (funding_z_val < -2.0) and vol_filter and (close_val > ema_50_val)
        short_entry = (funding_z_val > 2.0) and vol_filter and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when funding normalizes (Z > -0.5) or trend reversal
            if funding_z_val > -0.5 or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when funding normalizes (Z < 0.5) or trend reversal
            if funding_z_val < 0.5 or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_FundingRate_ZScore_MeanReversion_1wTrendFilter"
timeframe = "1d"
leverage = 1.0