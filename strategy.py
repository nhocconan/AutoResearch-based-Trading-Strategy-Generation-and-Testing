#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 1w volume regime filter.
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND 1w volume > 1.2 * 20-period average volume.
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND 1w volume > 1.2 * 20-period average volume.
# Exit when price crosses Camarilla H3/L3 levels (mean reversion within the day) or when trend reverses.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by combining intraday breakout structure with higher-timeframe trend and volume confirmation.
# Target: 60-100 total trades over 4 years (15-25/year) for 6h timeframe.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_1wVolumeRegime_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # needed for Camarilla calculation
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w volume regime filter (HTF) - additional delay not needed for EMA/volume averages
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1w > (1.2 * vol_ma_20_1w)  # above average volume regime
    volume_regime_aligned = align_htf_to_ltf(prices, df_1w, volume_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for Camarilla calculation
        # Need at least 1d of prior data for Camarilla (using prior day's OHLC)
        # We'll use the prior completed day's data - we need to get HTF data for the prior day
        # But since we're on 6h timeframe, we can calculate Camarilla levels from prior 1d bar
        # However, we don't have easy access to prior 1d bar values in the loop without look-ahead
        # Instead, we'll calculate Camarilla levels for the CURRENT day using current bar's open/high/low/close
        # But this would be look-ahead! So we must use PRIOR day's data
        
        # Simpler approach: use the 1d HTF data we already loaded, and use its completed bars
        # We need to get the prior completed 1d bar's OHLC for Camarilla calculation
        # Since we have df_1d from get_htf_data, we can access its values
        # But we need to align it properly to know which 1d bar is completed
        
        # Let's change approach: calculate Camarilla levels from 1d data and align them
        # This is cleaner and avoids look-ahead
        
        # We'll do this outside the loop for efficiency
        pass  # We'll move Camarilla calculation outside the loop
    
    # Let's restart with proper MTF approach for Camarilla levels
    
    # Calculate Camarilla levels from prior 1d bar
    # We need the prior completed 1d bar's OHLC
    # Since df_1d contains historical 1d bars, we can calculate Camarilla for each bar
    # then shift by 1 to use prior bar's levels, then align to 6h timeframe
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar using that bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Camarilla levels calculation
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # R2 = close + ((high - low) * 1.1 / 6)
    # R1 = close + ((high - low) * 1.1 / 12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1 / 12)
    # S2 = close - ((high - low) * 1.1 / 6)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    
    range_1d = high_1d - low_1d
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    h3_1d = close_1d + (range_1d * 1.1 / 6)  # H3 is same as R2
    l3_1d = close_1d - (range_1d * 1.1 / 6)  # L3 is same as S2
    
    # We want to use PRIOR day's Camarilla levels for today's trading
    # So shift the levels by 1 bar
    r3_1d_prior = np.roll(r3_1d, 1)
    s3_1d_prior = np.roll(s3_1d, 1)
    h3_1d_prior = np.roll(h3_1d, 1)
    l3_1d_prior = np.roll(l3_1d, 1)
    # First bar will have invalid prior data (rolled from last bar), set to nan
    r3_1d_prior[0] = np.nan
    s3_1d_prior[0] = np.nan
    h3_1d_prior[0] = np.nan
    l3_1d_prior[0] = np.nan
    
    # Align prior day's Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_prior)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_prior)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d_prior)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d_prior)
    
    # Recalculate 1d EMA34 alignment (we already have close_1d above)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Recalculate 1w volume regime alignment
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1w > (1.2 * vol_ma_20_1w)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1w, volume_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND close > 1d EMA34 AND volume regime active
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_regime_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 AND close < 1d EMA34 AND volume regime active
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_regime_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below H3 (mean reversion) OR trend reverses (close < EMA)
            if close[i] < h3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above L3 (mean reversion) OR trend reverses (close > EMA)
            if close[i] > l3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals