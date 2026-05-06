#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with 1d volume confirmation and 1w ATR volatility filter
# - Uses 1w Donchian channel (20-period) to identify structural breaks
# - Requires 1d volume > 1.5x 20-period average for confirmation
# - Uses 1w ATR(14) to filter low volatility environments (ATR < 50th percentile)
# - Enters long when price breaks above 1w upper Donchian with volume confirmation in low vol
# - Enters short when price breaks below 1w lower Donchian with volume confirmation in low vol
# - Exits when price returns to 1w midline or volatility expands (ATR > 80th percentile)
# - Designed to capture sustained moves after weekly consolidation with daily volume confirmation
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_1wDonchian_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels and ATR calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w Donchian Channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Upper band: highest high over 20 periods
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1w ATR (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Wilder's ATR smoothing
    def wilders_atr(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_atr(tr, 14)
    
    # Calculate 1w ATR percentile rank (lookback 50 periods)
    atr_series = pd.Series(atr_1w)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1w indicators to 1d timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_1d = align_htf_to_ltf(prices, df_1w, donchian_mid)
    atr_percentile_1d = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Volume filters (1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(donchian_mid_1d[i]) or np.isnan(atr_percentile_1d[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility environment (ATR < 50th percentile)
            low_vol = atr_percentile_1d[i] < 50
            
            if low_vol:
                # Long: price breaks above 1w upper Donchian with volume confirmation
                if close[i] > donchian_high_1d[i] and volume_confirmed[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1w lower Donchian with volume confirmation
                elif close[i] < donchian_low_1d[i] and volume_confirmed[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midline OR volatility expands (ATR > 80th percentile)
            if close[i] < donchian_mid_1d[i] or atr_percentile_1d[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midline OR volatility expands (ATR > 80th percentile)
            if close[i] > donchian_mid_1d[i] or atr_percentile_1d[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals