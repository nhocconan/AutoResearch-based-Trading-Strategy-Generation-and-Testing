#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Combined Momentum with Volume and Session Filter
# Uses 4h EMA(20) for trend direction, 1d VWAP deviation for mean reversion entry on 1h
# Volume spike (2x 20-period average) confirms momentum, session filter (08-20 UTC) reduces noise
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# Works in bull/bear: trend filter avoids counter-trend trades, VWAP deviation captures pullbacks

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA(20) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d VWAP for mean reversion reference
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    tpv_1d = typical_price_1d * df_1d['volume'].values
    cum_tpv_1d = np.nancumsum(tpv_1d)
    cum_vol_1d = np.nancumsum(df_1d['volume'].values)
    vwap_1d = np.full_like(typical_price_1d, np.nan)
    valid_vol_1d = cum_vol_1d != 0
    vwap_1d[valid_vol_1d] = cum_tpv_1d[valid_vol_1d] / cum_vol_1d[valid_vol_1d]
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 1h VWAP deviation (20-period)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.full_like(typical_price, np.nan)
    valid_vol = cum_vol != 0
    vwap[valid_vol] = cum_tpv[valid_vol] / cum_vol[valid_vol]
    
    price_dev = typical_price - vwap
    dev_series = pd.Series(price_dev)
    std_dev = dev_series.rolling(window=20, min_periods=20).std().values
    
    # Volume spike: 2x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 20  # for VWAP, std dev, EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(std_dev[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: 4h EMA direction
        above_ema_4h = price > ema_4h_aligned[i]
        
        # Mean reversion signal: price deviated from 1h VWAP
        dev_from_vwap = price - vwap[i]
        z_score = dev_from_vwap / std_dev[i] if std_dev[i] > 0 else 0
        
        if position == 0:
            # Long: oversold (below VWAP) in uptrend with volume spike
            if (z_score < -1.5) and above_ema_4h and vol_spike[i]:
                position = 1
                signals[i] = position_size
            # Short: overbought (above VWAP) in downtrend with volume spike
            elif (z_score > 1.5) and (not above_ema_4h) and vol_spike[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or trend changes
            if (price >= vwap[i]) or (price < ema_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP or trend changes
            if (price <= vwap[i]) or (price > ema_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h1d_Momentum_MeanRev_Volume_Session"
timeframe = "1h"
leverage = 1.0