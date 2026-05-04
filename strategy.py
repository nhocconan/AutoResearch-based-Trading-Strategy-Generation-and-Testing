#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX regime filter and volume confirmation
# In trending markets (1d ADX > 25): breakout in trend direction at Camarilla levels
# In ranging markets (1d ADX <= 25): mean reversion at Camarilla H4/L4 levels
# Volume confirmation (>1.5x 20-period EMA) filters low-quality signals
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.
# Strategy adapts to bull/bear markets via regime filter and uses 4h primary timeframe.

name = "4h_Camarilla_R3S3_1dADX_Regime_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) with proper min_periods
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    plus_dm = high_1d.diff()
    minus_dm = low_1d.diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_1d.sub(low_1d)
    tr2 = high_1d.sub(close_1d.shift(1)).abs()
    tr3 = low_1d.sub(close_1d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 1d ADX to 4h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    #            H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    #            H2 = close + 1.1*(high-low)/6, L2 = close - 1.1*(high-low)/6
    #            H1 = close + 1.1*(high-low)/12, L1 = close - 1.1*(high-low)/12
    # We'll use H3/L3 for breakouts and H4/L4 for stronger signals
    # For simplicity, we use the typical Camarilla calculation based on previous day
    
    # Calculate typical price for Camarilla (using previous 1d bar)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    H4 = typical_price_1d + 1.1 * range_1d / 2
    H3 = typical_price_1d + 1.1 * range_1d / 4
    L3 = typical_price_1d - 1.1 * range_1d / 4
    L4 = typical_price_1d - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4.values)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3.values)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3.values)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4.values)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            if adx_aligned[i] > 25:
                # Trending market: breakout in trend direction at H3/L3
                # Determine trend direction using 1d +DI/-DI (simplified: use price vs EMA)
                # We'll use a simple trend filter: price above/below 20-period EMA on 1d
                # For efficiency, we'll use price vs midpoint of H3/L3 as trend proxy
                midpoint = (H3_aligned[i] + L3_aligned[i]) / 2
                if close[i] > midpoint:
                    # Uptrend bias: long on break above H3
                    if close[i] > H3_aligned[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                else:
                    # Downtrend bias: short on break below L3
                    if close[i] < L3_aligned[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: mean reversion at H4/L4 (stronger levels)
                if close[i] <= L4_aligned[i] and volume_confirm:
                    # Long at lower band (L4)
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= H4_aligned[i] and volume_confirm:
                    # Short at upper band (H4)
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint OR ADX weakens (<20) OR volume drops
            midpoint = (H3_aligned[i] + L3_aligned[i]) / 2
            if (close[i] <= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR ADX weakens (<20) OR volume drops
            midpoint = (H3_aligned[i] + L3_aligned[i]) / 2
            if (close[i] >= midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals