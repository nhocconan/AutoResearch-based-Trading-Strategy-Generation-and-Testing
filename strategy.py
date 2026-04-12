#!/usr/bin/env python3
"""
4h_1d_Bollinger_Band_Mean_Reversion_v1
Hypothesis: In ranging markets, price reverses at Bollinger Band extremes (2 std dev) with confirmation from RSI extremes and volume contraction. Uses 1d trend filter to avoid counter-trend trades. Works in both bull and bear markets by adapting to regime (uses Bollinger Band Width percentile to detect ranging vs trending). Target: 20-30 trades per year (80-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Bollinger_Band_Mean_Reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for trend filter and BBW percentile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # === 4H BOLLINGER BANDS (20, 2) ===
    bb_period = 20
    bb_std = 2.0
    sma = np.full(n, np.nan)
    bb_std_dev = np.full(n, np.nan)
    
    if n >= bb_period:
        # Calculate SMA
        sma_sum = np.sum(close[:bb_period])
        sma[bb_period-1] = sma_sum / bb_period
        for i in range(bb_period, n):
            sma_sum = sma_sum - close[i-bb_period] + close[i]
            sma[i] = sma_sum / bb_period
        
        # Calculate standard deviation
        for i in range(bb_period-1, n):
            bb_std_dev[i] = np.std(close[i-bb_period+1:i+1])
    
    upper_band = sma + (bb_std * bb_std_dev)
    lower_band = sma - (bb_std * bb_std_dev)
    
    # === 1D RSI (14) FOR TREND FILTER ===
    delta = pd.Series(daily_close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1D BOLLINGER BAND WIDTH PERCENTILE (50-period) FOR REGIME DETECTION ===
    bbw_period = 50
    bbw_lookback = 50
    sma_1d = np.full(len(daily_close), np.nan)
    bb_std_dev_1d = np.full(len(daily_close), np.nan)
    bb_width = np.full(len(daily_close), np.nan)
    
    if len(daily_close) >= bbw_period:
        # Calculate 1d SMA
        sma_sum = np.sum(daily_close[:bbw_period])
        sma_1d[bbw_period-1] = sma_sum / bbw_period
        for i in range(bbw_period, len(daily_close)):
            sma_sum = sma_sum - daily_close[i-bbw_period] + daily_close[i]
            sma_1d[i] = sma_sum / bbw_period
        
        # Calculate 1d standard deviation and BB Width
        for i in range(bbw_period-1, len(daily_close)):
            bb_std_dev_1d[i] = np.std(daily_close[i-bbw_period+1:i+1])
            bb_width[i] = ( (sma_1d[i] + (bb_std * bb_std_dev_1d[i])) - (sma_1d[i] - (bb_std * bb_std_dev_1d[i])) ) / sma_1d[i]
    
    # Calculate percentile of current BBW over lookback period
    bbw_percentile = np.full(len(daily_close), np.nan)
    if len(daily_close) >= bbw_period + bbw_lookback:
        for i in range(bbw_period + bbw_lookback - 1, len(daily_close)):
            lookback_data = bb_width[i-bbw_lookback+1:i+1]
            valid_data = lookback_data[~np.isnan(lookback_data)]
            if len(valid_data) > 0:
                current_bw = bb_width[i]
                if not np.isnan(current_bw) and sma_1d[i] != 0:
                    bbw_percentile[i] = (np.sum(valid_data <= current_bw) / len(valid_data)) * 100
    
    bbw_percentile_aligned = align_htf_to_ltf(prices, df_1d, bbw_percentile)
    
    # === VOLUME CONTRACTION (below 50% of 20-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_contraction = volume < (vol_ma * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(bbw_percentile_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade in ranging markets (BBW percentile between 30 and 70)
        is_ranging = (bbw_percentile_aligned[i] >= 30) and (bbw_percentile_aligned[i] <= 70)
        
        # Trend filter: avoid strong trends (RSI not too extreme)
        rsi_not_extreme = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Mean reversion conditions
        at_lower_band = low[i] <= lower_band[i]
        at_upper_band = high[i] >= upper_band[i]
        rsi_oversold = rsi_1d_aligned[i] < 30  # Using 1d RSI for mean reversion signal
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Entry conditions: BB touch + RSI extreme + volume contraction + ranging regime
        long_entry = at_lower_band and rsi_oversold and vol_contraction[i] and is_ranging and rsi_not_extreme
        short_entry = at_upper_band and rsi_overbought and vol_contraction[i] and is_ranging and rsi_not_extreme
        
        # Exit conditions: price returns to middle band or opposite signal
        long_exit = close[i] >= sma[i]
        short_exit = close[i] <= sma[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals