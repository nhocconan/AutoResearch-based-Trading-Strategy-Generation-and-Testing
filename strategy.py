#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume confirmation and 1d ADX trend filter
# Bollinger Band width < 50th percentile indicates low volatility (squeeze). Breakout above upper band
# or below lower band with volume surge signals strong directional move. 1d ADX > 25 ensures we only
# trade in strong trends, avoiding whipsaws in ranges. This strategy captures volatility expansion
# phases that occur in both bull and bear markets. Targets 15-25 trades per year (~60-100 total over 4 years)
# to minimize fee drag while maintaining edge in volatile markets.

name = "6h_BollingerSqueeze_1dVolume_1dADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = ma + (std * bb_std)
    lower_band = ma - (std * bb_std)
    bb_width = (upper_band - lower_band) / ma
    
    # Bollinger Band width percentile (lookback 50 periods ~ 150h)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True) * 100
    squeeze = bb_width_percentile < 50  # Below 50th percentile = squeeze
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Volume spike detection on 1d (24-period MA = 4 days)
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=24, min_periods=24).mean()
    vol_spike = df_1d['volume'].values > (vol_ma.values * 2.0)
    vol_spike_6h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # ADX calculation on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr = np.zeros_like(high_1d)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
        
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
    
    # Wilder smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_6h = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_6h = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 24)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(squeeze[i]) or np.isnan(vol_spike_6h[i]) or 
            np.isnan(adx_strong_6h[i]) or np.isnan(adx_weak_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: breakout above upper band during squeeze, volume spike, strong trend
            if close[i] > upper_band[i] and squeeze[i] and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below lower band during squeeze, volume spike, strong trend
            elif close[i] < lower_band[i] and squeeze[i] and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle band or trend weakens
            if close[i] < ma[i] or adx_weak_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle band or trend weakens
            if close[i] > ma[i] or adx_weak_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals