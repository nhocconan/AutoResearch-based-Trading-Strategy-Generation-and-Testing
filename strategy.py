#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 12h EMA Trend Filter + Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA13. In trending markets (12h EMA50 slope),
# we take trades in trend direction: long when Bear Power < 0 & Bull Power rising, short when Bull Power > 0 & Bear Power falling.
# In ranging markets (ADX<25 on 12h), we fade extremes: long when Bull Power < -std, short when Bear Power > +std.
# Volume spike (>1.5x 20-period EMA) confirms momentum. Designed for 6h timeframe targeting 75-175 total trades.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_ElderRay_12hEMA_Trend_ADXRegime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and regime filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = np.diff(ema_50_12h, prepend=ema_50_12h[0])  # slope of EMA50
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_50_slope)
    
    # Calculate 12h ADX (14-period) for regime detection
    plus_dm = pd.Series(df_12h['high']).diff()
    minus_dm = pd.Series(df_12h['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_12h['high']).sub(df_12h['low'])
    tr2 = pd.Series(df_12h['high']).sub(df_12h['close'].shift(1)).abs()
    tr3 = pd.Series(df_12h['low']).sub(df_12h['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_12h = dx.rolling(window=14, min_periods=14).mean()
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h.values)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_slope_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: trending (ADX>=25) or ranging (ADX<25)
            if adx_12h_aligned[i] >= 25:
                # Trending market: trade with 12h EMA50 slope
                if ema_50_slope_aligned[i] > 0:  # Uptrend
                    # Long: Bear Power < 0 (price below EMA13) AND Bull Power rising (momentum building)
                    if (bear_power[i] < 0 and 
                        bull_power[i] > bull_power[i-1] and 
                        volume_confirm):
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend (ema_50_slope_aligned[i] <= 0)
                    # Short: Bull Power > 0 (price above EMA13) AND Bear Power falling (momentum building)
                    if (bull_power[i] > 0 and 
                        bear_power[i] < bear_power[i-1] and 
                        volume_confirm):
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: mean reversion at extremes
                # Calculate volatility-adjusted thresholds
                bp_std = np.nanstd(bear_power[max(0, i-50):i+1]) if i >= 50 else np.nanstd(bear_power[:i+1])
                bull_std = np.nanstd(bull_power[max(0, i-50):i+1]) if i >= 50 else np.nanstd(bull_power[:i+1])
                
                # Avoid division by zero or NaN
                if np.isnan(bp_std) or bp_std == 0:
                    bp_std = 1.0
                if np.isnan(bull_std) or bull_std == 0:
                    bull_std = 1.0
                
                # Long: Bull Power < -0.5 * std (oversold)
                if (bull_power[i] < (-0.5 * bull_std) and volume_confirm):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > +0.5 * std (overbought)
                elif (bear_power[i] > (0.5 * bp_std) and volume_confirm):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Bull Power > 0 (price above EMA13) OR ADX weakening (<20) OR volume drops
            if (bull_power[i] > 0 or 
                adx_12h_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power < 0 (price below EMA13) OR ADX weakening (<20) OR volume drops
            if (bear_power[i] < 0 or 
                adx_12h_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals