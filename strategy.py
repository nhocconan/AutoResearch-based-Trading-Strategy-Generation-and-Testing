#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with weekly ADX trend filter and volume confirmation.
# Bollinger Band squeeze indicates low volatility, often preceding breakout moves.
# Weekly ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Volume confirmation ensures breakouts have institutional participation.
# Designed for very low trade frequency (5-15/year) to minimize fee drag in 1d timeframe.
# Works in bull markets (breakouts continue with trend) and bear markets (breakdowns continue with trend).
name = "1d_BollingerSqueeze_ADXTrend_Volume"
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
    
    # Get daily data for Bollinger Bands (primary timeframe data)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Bollinger Bands (20, 2)
    close_series = pd.Series(df_1d['close'])
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width (normalized by middle band)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Bollinger Band Squeeze: BB Width at 20-period low (indicates low volatility)
    bb_width_low = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    squeeze_condition = bb_width <= bb_width_low * 1.1  # Within 10% of recent low
    
    # Calculate weekly ADX for trend filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values[i] / atr[i])
                minus_di[i] = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values[i] / atr[i])
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_14_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for all indicators to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(squeeze_condition[i]) or
            np.isnan(adx_14_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trending = adx_14_1w_aligned[i] > 25
        
        if not trending:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper BB during squeeze with volume
            if vol_confirm and close[i] > bb_upper[i] and squeeze_condition[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB during squeeze with volume
            elif vol_confirm and close[i] < bb_lower[i] and squeeze_condition[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle BB (mean reversion) or squeeze ends
            if close[i] < bb_middle[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle BB or squeeze ends
            if close[i] > bb_middle[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals