#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 20-period EMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA20 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Get daily data for Parabolic SAR (trend direction)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_step = 0.02
    af_max = 0.2
    
    # Initialize SAR arrays
    psar = np.zeros(len(high_1d))
    trend = np.ones(len(high_1d))  # 1 for uptrend, -1 for downtrend
    af = np.zeros(len(high_1d))
    ep = np.zeros(len(high_1d))
    
    # Initialize first values
    psar[0] = low_1d[0]
    ep[0] = high_1d[0]
    af[0] = af_start
    trend[0] = 1
    
    # Calculate Parabolic SAR
    for i in range(1, len(high_1d)):
        if trend[i-1] == 1:  # uptrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            if low_1d[i] < psar[i]:  # trend reversal
                trend[i] = -1
                psar[i] = ep[i-1]
                ep[i] = low_1d[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high_1d[i] > ep[i-1]:
                    ep[i] = high_1d[i]
                    af[i] = min(af[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            if high_1d[i] > psar[i]:  # trend reversal
                trend[i] = 1
                psar[i] = ep[i-1]
                ep[i] = high_1d[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low_1d[i] < ep[i-1]:
                    ep[i] = low_1d[i]
                    af[i] = min(af[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Align Parabolic SAR trend to 6h timeframe
    psar_trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Get 12h data for Donchian channel (price channel)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian(10) channel
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    donch_high_10 = high_12h_series.rolling(window=10, min_periods=10).max().values
    donch_low_10 = low_12h_series.rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 12h timeframe
    donch_high_10_aligned = align_htf_to_ltf(prices, df_12h, donch_high_10)
    donch_low_10_aligned = align_htf_to_ltf(prices, df_12h, donch_low_10)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # ATR(10) for volatility filter and stop
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(psar_trend_aligned[i]) or 
            np.isnan(donch_high_10_aligned[i]) or np.isnan(donch_low_10_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high with volume, daily EMA20 uptrend, and PSAR uptrend
            if (close[i] > donch_high_10_aligned[i] and volume_filter[i] and 
                close[i] > ema20_1d_aligned[i] and psar_trend_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low with volume, daily EMA20 downtrend, and PSAR downtrend
            elif (close[i] < donch_low_10_aligned[i] and volume_filter[i] and 
                  close[i] < ema20_1d_aligned[i] and psar_trend_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low OR ATR-based stop
            if close[i] < donch_low_10_aligned[i] or close[i] < (high[max(0, i-1)] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high OR ATR-based stop
            if close[i] > donch_high_10_aligned[i] or close[i] > (low[max(0, i-1)] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA20_PSAR_12h_Donchian10_Volume"
timeframe = "6h"
leverage = 1.0