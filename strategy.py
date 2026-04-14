#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX Trend Strength + Volume Surge + Price Reversion to VWAP
# Uses ADX(14) > 25 to identify strong trends, then enters on pullbacks to VWAP
# Volume surge (current volume > 1.5x 20-period average) confirms momentum
# Works in both bull/bear markets by only trading with strong trend direction
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate VWAP (20-period) on 12h data
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.full_like(typical_price, np.nan)
    valid_vol = cum_vol != 0
    vwap[valid_vol] = cum_tpv[valid_vol] / cum_vol[valid_vol]
    
    # Standard deviation of price from VWAP (20-period)
    price_dev = typical_price - vwap
    dev_series = pd.Series(price_dev)
    std_dev = dev_series.rolling(window=20, min_periods=20).std().values
    
    # Volume surge: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for ADX and VWAP calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(std_dev[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_1w_aligned[i] > 25
        
        if not strong_trend:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to VWAP from below in uptrend with volume surge
            if (price < vwap[i] - 0.5 * std_dev[i] and 
                price > vwap[i] - 2.0 * std_dev[i] and
                volume_surge[i]):
                position = 1
                signals[i] = position_size
            # Short: price pulls back to VWAP from above in downtrend with volume surge
            elif (price > vwap[i] + 0.5 * std_dev[i] and 
                  price < vwap[i] + 2.0 * std_dev[i] and
                  volume_surge[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches VWAP or trend weakens
            if price >= vwap[i] or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches VWAP or trend weakens
            if price <= vwap[i] or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_ADX_Trend_VWAP_Pullback"
timeframe = "12h"
leverage = 1.0