#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and ATR regime filter
# - Long when price breaks above Camarilla H3 level AND 12h volume > 1.8x 20-period average AND 12h ATR(14) < median ATR(20)
# - Short when price breaks below Camarilla L3 level AND 12h volume > 1.8x 20-period average AND 12h ATR(14) < median ATR(20)
# - Exit when price crosses back inside Camarilla H3-L3 range
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels with high probability reaction
# - Volume confirmation reduces false breakouts
# - ATR regime filter ensures we trade during low volatility when breakouts are more sustainable

name = "4h_12h_camarilla_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Camarilla pivot levels (based on previous day's OHLC)
    # We need daily data for pivot calculation, resample 4h to daily OHLC
    close_ser = pd.Series(close)
    high_ser = pd.Series(high)
    low_ser = pd.Series(low)
    
    # Calculate daily OHLC from 4h data (assuming 6x 4h bars per day)
    daily_high = high_ser.rolling(window=6, min_periods=6).max().shift(6)  # Previous day's high
    daily_low = low_ser.rolling(window=6, min_periods=6).min().shift(6)    # Previous day's low
    daily_close = close_ser.rolling(window=6, min_periods=6).last().shift(6) # Previous day's close
    
    # Camarilla levels: H4 = C + ((H-L)*1.1/2), H3 = C + ((H-L)*1.1/4), etc.
    # We focus on H3 and L3 for breakout trading
    daily_range = daily_high - daily_low
    camarilla_h3 = daily_close + (daily_range * 1.1 / 4)
    camarilla_l3 = daily_close - (daily_range * 1.1 / 4)
    
    # Forward fill to handle NaN values from rolling window
    camarilla_h3 = pd.Series(camarilla_h3).ffill().bfill().values
    camarilla_l3 = pd.Series(camarilla_l3).ffill().bfill().values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 12h ATR(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_12h = np.zeros_like(tr)
    atr_12h[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_median_20 = pd.Series(atr_12h).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_12h < atr_median_20
    
    # Align HTF indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_12h, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(low_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND low volatility regime
            if (close[i] > camarilla_h3[i] and 
                volume_spike_aligned[i] and 
                low_vol_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND low volatility regime
            elif (close[i] < camarilla_l3[i] and 
                  volume_spike_aligned[i] and 
                  low_vol_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside Camarilla H3-L3 range
            exit_long = (position == 1 and close[i] < camarilla_h3[i])
            exit_short = (position == -1 and close[i] > camarilla_l3[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals