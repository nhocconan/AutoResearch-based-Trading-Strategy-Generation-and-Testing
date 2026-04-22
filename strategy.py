#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Doji breakout with 4h VWAP trend filter and volume spike
    # Doji (open≈close) indicates indecision, breakout from Doji range signals new momentum
    # 4h VWAP acts as dynamic support/resistance, filters for institutional trend direction
    # Volume spike (2x 20-period MA) confirms participation, works in bull/bear via breakout direction
    # Session filter (08-20 UTC) reduces noise, target 15-35 trades/year
    
    # Price and volume data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Doji range (body size)
    body_size = np.abs(close - open_)
    avg_body = pd.Series(body_size).rolling(window=20, min_periods=20).mean().values
    is_doji = body_size < 0.1 * avg_body  # Doji: body < 10% of average body
    
    # Track Doji high/low for breakout
    doji_high = np.where(is_doji, high, np.nan)
    doji_low = np.where(is_doji, low, np.nan)
    
    # Forward fill Doji levels until broken
    doji_high_series = pd.Series(doji_high)
    doji_low_series = pd.Series(doji_low)
    doji_high_ffill = doji_high_series.ffill().values
    doji_low_ffill = doji_low_series.ffill().values
    
    # Load 4h data for VWAP trend
    df_4h = get_htf_data(prices, '4h')
    # Calculate VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    vwap = (typical_price * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_values = vwap.values
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_values)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(doji_high_ffill[i]) or 
            np.isnan(doji_low_ffill[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Doji high with volume spike and price above 4h VWAP (uptrend)
            if high[i] > doji_high_ffill[i] and vol_spike[i] and close[i] > vwap_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below Doji low with volume spike and price below 4h VWAP (downtrend)
            elif low[i] < doji_low_ffill[i] and vol_spike[i] and close[i] < vwap_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Return to opposite Doji level
            if position == 1:
                if low[i] < doji_low_ffill[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if high[i] > doji_high_ffill[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Doji_Breakout_4hVWAP_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0