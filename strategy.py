#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume spike confirmation
# Long when Williams %R(14) crosses above -80 (oversold) + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period avg
# Short when Williams %R(14) crosses below -20 (overbought) + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Williams %R identifies exhaustion points; 1d EMA34 ensures we trade with higher-timeframe trend.
# Volume spike confirms institutional participation at turning points. Works in ranging and trending markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data for Williams %R calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # === 6h Indicator: Williams %R(14) ===
    highest_high_14 = pd.Series(df_6h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_6h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - df_6h['close'].values) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 (trend filter) ===
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period SMA on 6h volume
    vol_sma_20_6h = pd.Series(df_6h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_sma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_aligned[i] * 2.0)
        
        # Williams %R cross above -80 (bullish reversal from oversold)
        wr_cross_up = (williams_r_aligned[i] > -80) and (williams_r_aligned[i-1] <= -80)
        # Williams %R cross below -20 (bearish reversal from overbought)
        wr_cross_down = (williams_r_aligned[i] < -20) and (williams_r_aligned[i-1] >= -20)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (exiting oversold)
        # 2. Price above 1d EMA34 (uptrend filter)
        # 3. Volume confirmation
        if wr_cross_up and (close[i] > ema_34_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (exiting overbought)
        # 2. Price below 1d EMA34 (downtrend filter)
        # 3. Volume confirmation
        elif wr_cross_down and (close[i] < ema_34_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_EMA34_VolumeSpike_Filter_v1"
timeframe = "6h"
leverage = 1.0