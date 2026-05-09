# 1d_GoldenCross_VolumeConfirm_1wTrend
# Hypothesis: Daily golden cross (SMA50 > SMA200) with volume confirmation and weekly trend filter.
# Golden cross indicates long-term bullish momentum; volume > 1.5x average confirms institutional participation.
# Weekly EMA50 trend filter ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (<25/year) to minimize fee drag in ranging/bear markets.
# Works in bull markets via trend continuation and in bear markets via mean-reversion pulls to support.

name = "1d_GoldenCross_VolumeConfirm_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily SMA50 and SMA200 for golden cross
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Calculate 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Need 200 for SMA200 and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        sma50 = sma_50[i]
        sma200 = sma_200[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        weekly_ema = ema_50_1w_aligned[i]
        
        if position == 0:
            # Enter long: Golden cross (SMA50 > SMA200) AND volume > 1.5x average AND price above weekly EMA (uptrend)
            if sma50 > sma200 and vol > 1.5 * vol_ma and close[i] > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Enter short: Death cross (SMA50 < SMA200) AND volume > 1.5x average AND price below weekly EMA (downtrend)
            elif sma50 < sma200 and vol > 1.5 * vol_ma and close[i] < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Death cross OR price below weekly EMA (trend reversal)
            if sma50 < sma200 or close[i] < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Golden cross OR price above weekly EMA (trend reversal)
            if sma50 > sma200 or close[i] > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals