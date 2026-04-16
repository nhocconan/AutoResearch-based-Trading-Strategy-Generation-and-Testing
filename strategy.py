#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and volume spike confirmation.
# Long when price < 4h VWAP (oversold) AND 1d EMA50 uptrend (price > EMA50) AND 1h volume > 2x 20-period average.
# Short when price > 4h VWAP (overbought) AND 1d EMA50 downtrend (price < EMA50) AND 1h volume > 2x 20-period average.
# Uses discrete position size 0.20. 4h VWAP identifies mean reversion levels, 1d EMA50 ensures alignment with higher timeframe trend,
# volume spike confirms participation. Designed to work in both bull (buy dips) and bear (sell rallies) markets.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h. Session filter 08-20 UTC reduces noise.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Volume Spike (volume > 2x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === 1h Indicators: Typical Price for VWAP ===
    typical_price = (high + low + close) / 3.0
    tp_volume = typical_price * volume
    
    # Get 4h data once before loop for VWAP
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough for VWAP calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    tp_volume_4h = typical_price_4h * volume_4h
    
    # === 4h Indicators: VWAP (cumulative, reset daily) ===
    # Since we don't have daily reset in HTF data, use rolling 24-period (6h * 4 = 24h) as proxy
    cum_tp_vol_4h = pd.Series(tp_volume_4h).rolling(window=24, min_periods=24).sum().values
    cum_vol_4h = pd.Series(volume_4h).rolling(window=24, min_periods=24).sum().values
    vwap_4h = np.divide(cum_tp_vol_4h, cum_vol_4h, out=np.zeros_like(cum_tp_vol_4h), where=cum_vol_4h!=0)
    
    # Align 4h VWAP to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 24 for VWAP, 20 for volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vwap_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vwap_4h = vwap_4h_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price rises above 4h VWAP (mean reversion complete) or volume spike ends
            if price > vwap_4h or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price falls below 4h VWAP (mean reversion complete) or volume spike ends
            if price < vwap_4h or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price < 4h VWAP (oversold) AND price > 1d EMA50 (uptrend) AND volume spike
            if price < vwap_4h and price > ema_1d and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: price > 4h VWAP (overbought) AND price < 1d EMA50 (downtrend) AND volume spike
            elif price > vwap_4h and price < ema_1d and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_VWAP_MeanReversion_1dEMA50_VolumeSpike_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0