#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 1d volume-weighted average price (VWAP) as dynamic support/resistance,
# filtered by 1w trend direction. Price tends to revert to VWAP in ranging markets and break
# with volume in trending markets. Works in bull/bear via 1w trend filter.
# Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP = sum(price * volume) / sum(volume)
    # Using typical price = (H+L+C)/3
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vp_1d = typical_price_1d * df_1d['volume']
    cum_vp_1d = vp_1d.cumsum().values
    cum_vol_1d = df_1d['volume'].cumsum().values
    vwap_1d = np.where(cum_vol_1d != 0, cum_vp_1d / cum_vol_1d, typical_price_1d.values)
    
    # Align VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h ATR(20) for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 4h volume ratio (current / 50-period average)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.where(vol_ma_50 == 0, 1, vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_20[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        atr = atr_20[i]
        vol_ratio_4h = vol_ratio[i]
        
        # Determine market regime from weekly trend
        uptrend = price > ema_trend
        downtrend = price < ema_trend
        
        # Volatility filter: avoid extreme volatility
        atr_ma_50 = pd.Series(atr_20).rolling(window=50, min_periods=50).mean().values[i]
        vol_filter = (atr < 3.0 * atr_ma_50)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_4h > 1.3)
        
        if position == 0:
            # In uptrend: look for long near VWAP (support)
            if uptrend and vol_filter:
                if price <= vwap * 1.005:  # Near VWAP with small buffer
                    signals[i] = 0.25
                    position = 1
            # In downtrend: look for short near VWAP (resistance)
            elif downtrend and vol_filter:
                if price >= vwap * 0.995:  # Near VWAP with small buffer
                    signals[i] = -0.25
                    position = -1
            # In ranging (no clear trend): fade extremes
            else:
                if price <= vwap * 1.005 and vol_filter:  # Near VWAP
                    signals[i] = 0.25
                    position = 1
                elif price >= vwap * 0.995 and vol_filter:  # Near VWAP
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches VWAP or stops reversed
            if price >= vwap or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches VWAP or stops reversed
            if price <= vwap or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_VWAP_MeanReversion_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0