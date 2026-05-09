#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PairsTrading_BTC_ETH_Zscore"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    Pairs trading strategy using BTC-ETH spread z-score.
    Long when spread is significantly undervalued (ETH cheap vs BTC).
    Short when spread is significantly overvalued (ETH expensive vs BTC).
    Uses 60-period z-score with entry at ±2.0 and exit at ±0.5.
    Works in both bull and bear markets as market-neutral strategy.
    """
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get BTC and ETH data for spread calculation
    # Note: This strategy assumes prices DataFrame contains ETHUSDT data
    # We need to load BTCUSDT data separately for the spread
    try:
        # Load BTC data (same timeframe)
        btc_prices = pd.read_parquet('data/cache/compressed/btcusdt.parquet')
        # Align to same index as ETH data
        btc_close = btc_prices.set_index('open_time')['close']
        # Reindex to match ETH prices index
        btc_close_aligned = btc_close.reindex(prices.set_index('open_time')['close'].index, method='ffill').values
    except:
        # Fallback: if BTC data not available, return zeros
        return np.zeros(n)
    
    eth_close = prices['close'].values
    
    # Calculate log spread: log(ETH) - log(BTC)
    spread = np.log(eth_close) - np.log(btc_close_aligned)
    
    # Calculate rolling z-score of spread (60-period)
    spread_series = pd.Series(spread)
    spread_mean = spread_series.rolling(window=60, min_periods=60).mean().values
    spread_std = spread_series.rolling(window=60, min_periods=60).std().values
    # Avoid division by zero
    zscore = np.where(spread_std > 0, (spread - spread_mean) / spread_std, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long spread (long ETH, short BTC), -1: short spread
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if np.isnan(zscore[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        z = zscore[i]
        
        if position == 0:
            # Enter long spread (ETH undervalued vs BTC)
            if z < -2.0:
                signals[i] = 0.25  # 25% position size
                position = 1
            # Enter short spread (ETH overvalued vs BTC)
            elif z > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long spread when z-score reverts to zero
            if z > -0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short spread when z-score reverts to zero
            if z < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals