#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume spike and 1d ADX regime filter
# - Primary: 1h price breaking above/below Camarilla H3/L3 levels captures intraday momentum
# - Volume filter: 4h volume > 1.5x 20-period volume MA confirms participation
# - Regime filter: 1d ADX(14) > 25 ensures strong trending market (avoids weak trends and ranging)
# - Exit: Price reverses back to Camarilla H4/L4 levels
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Session filter: 08-20 UTC to reduce noise trades
# - Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# - Works in bull/bear: Camarilla adapts to volatility, volume confirms strength, ADX filters weak trends

name = "1h_4h_1d_camarilla_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1h Camarilla pivot levels (based on previous day)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # We use daily high/low/close from 1d timeframe
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h volume spike filter: volume > 1.5x 20-period volume MA
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    
    # Calculate 1d ADX(14) for regime filter
    high_diff_1d = high_1d - np.roll(high_1d, 1)
    low_diff_1d = np.roll(low_1d, 1) - low_1d
    close_diff_1d = np.roll(close_1d, 1) - close_1d
    high_diff_1d[0] = 0
    low_diff_1d[0] = 0
    close_diff_1d[0] = 0
    
    plus_dm_1d = np.where((high_diff_1d > low_diff_1d) & (high_diff_1d > 0), high_diff_1d, 0)
    minus_dm_1d = np.where((low_diff_1d > high_diff_1d) & (low_diff_1d > 0), low_diff_1d, 0)
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = high_1d[0] - low_1d[0]
    tr2_1d[0] = np.abs(high_1d[0] - close_1d[0])
    tr3_1d[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_14_1d = pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).mean().values
    minus_dm_14_1d = pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).mean().values
    
    plus_di_1d = np.where(atr_14_1d > 0, 100 * plus_dm_14_1d / atr_14_1d, 0)
    minus_di_1d = np.where(atr_14_1d > 0, 100 * minus_dm_14_1d / atr_14_1d, 0)
    
    dx_1d = np.where((plus_di_1d + minus_di_1d) > 0, 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_ma_20_4h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period volume MA
        volume_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)
        vol_spike = volume_4h_current[i] > 1.5 * volume_ma_20_4h_aligned[i]
        
        # Regime filter: ADX > 25 to ensure strong trending conditions
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + vol spike + strong trend + session
            if (close[i] > camarilla_h3_aligned[i] and 
                vol_spike and strong_trend):
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below Camarilla L3 + vol spike + strong trend + session
            elif (close[i] < camarilla_l3_aligned[i] and 
                  vol_spike and strong_trend):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price reverses back to Camarilla H4/L4 levels
            if position == 1:  # Long position
                if close[i] < camarilla_h4_aligned[i]:  # Exit when price crosses below H4
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if close[i] > camarilla_l4_aligned[i]:  # Exit when price crosses above L4
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals