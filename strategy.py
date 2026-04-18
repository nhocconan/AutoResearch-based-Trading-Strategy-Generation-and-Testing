#!/usr/bin/env python3
"""
6h_MonthlyVWAP_Deviation_Reversion_v1
Hypothesis: Mean reversion to monthly VWAP with volume confirmation and ATR filter. 
In both bull and bear markets, price tends to revert to monthly VWAP after extended deviations. 
Enter long when price deviates >1.5σ below monthly VWAP with volume >1.5x average and ATR contraction.
Enter short when price deviates >1.5σ above monthly VWAP with volume >1.5x average and ATR contraction.
Exit when price returns to monthly VWAP or ATR expands indicating trend resumption.
Uses 1d VWAP calculated from intraday data (approximated via typical price * volume) and resampled to monthly.
Target: 15-25 trades/year via strict deviation threshold and volume/ATR filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    typical = (high + low + close) / 3.0
    
    # Get 1d data for monthly VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily typical price and volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate VWAP components: cumulative typical price * volume and cumulative volume
    tpv = typical_1d * volume_1d
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(volume_1d)
    
    # Calculate daily VWAP (avoid division by zero)
    vwap_1d = np.where(cum_vol > 0, cum_tpv / cum_vol, typical_1d)
    
    # Resample to monthly: take last VWAP of each month
    # Create date index for 1d data
    if len(df_1d) > 0:
        dates_1d = pd.to_datetime(df_1d['open_time'])
        # Find month ends
        month_ends = (dates_1d.to_series().dt.to_period('M') != 
                     dates_1d.to_series().shift(1).dt.to_period('M'))
        month_ends.iloc[0] = True  # First day is always a month start
        
        # Get monthly VWAP values (last VWAP of each month)
        monthly_vwap_raw = vwap_1d[month_ends.values]
        monthly_dates = dates_1d[month_ends.values]
        
        # Create monthly series and forward fill to daily
        if len(monthly_vwap_raw) > 0:
            monthly_series = pd.Series(monthly_vwap_raw, index=monthly_dates)
            # Reindex to daily frequency and forward fill
            daily_index = pd.date_range(start=dates_1d.iloc[0], 
                                       end=dates_1d.iloc[-1], 
                                       freq='D')
            monthly_vwap_daily = monthly_series.reindex(daily_index, method='ffill')
            # Align back to original daily index
            vwap_monthly = np.interp(np.arange(len(dates_1d)), 
                                   np.arange(len(daily_index)),
                                   monthly_vwap_daily.values)
        else:
            vwap_monthly = vwap_1d.copy()
    else:
        vwap_monthly = typical_1d.copy()
    
    # Align monthly VWAP to 6h timeframe
    vwap_monthly_aligned = align_htf_to_ltf(prices, df_1d, vwap_monthly)
    
    # Calculate deviation from monthly VWAP in ATR units
    # First calculate ATR(14) on 6h data
    atr_period = 14
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate price deviation from monthly VWAP
    deviation = close - vwap_monthly_aligned
    deviation_atr = np.where(atr > 0, deviation / atr, 0)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    if n >= vol_period:
        for i in range(vol_period, n):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # ATR contraction filter: current ATR < 0.8 * ATR(50) average
    atr_ma = np.full(n, np.nan)
    atr_ma_period = 50
    if n >= atr_ma_period:
        for i in range(atr_ma_period, n):
            atr_ma[i] = np.mean(atr[i - atr_ma_period:i])
    
    atr_contraction = (atr < 0.8 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, vol_period, atr_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_monthly_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        atr_contract = atr_contraction[i]
        
        if position == 0:
            # Long: price >1.5σ below monthly VWAP + volume + ATR contraction
            if deviation_atr[i] < -1.5 and vol_confirm and atr_contract:
                signals[i] = 0.25
                position = 1
            # Short: price >1.5σ above monthly VWAP + volume + ATR contraction
            elif deviation_atr[i] > 1.5 and vol_confirm and atr_contract:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP or ATR expansion (trend resumption)
            if deviation_atr[i] > -0.5 or not atr_contract:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or ATR expansion (trend resumption)
            if deviation_atr[i] < 0.5 or not atr_contract:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MonthlyVWAP_Deviation_Reversion_v1"
timeframe = "6h"
leverage = 1.0