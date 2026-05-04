#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d ADX regime filter and volume confirmation
# In low volatility squeeze (BB Width < 20th percentile): breakout in direction of 1d ADX trend
# Volume confirmation (>1.5x 20-period EMA) filters false breakouts
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 trades over 4 years.
# Works in bull/bear via ADX regime: ADX>25 = trend follow breakout, ADX<=25 = wait for stronger squeeze

name = "12h_BBSqueeze_1dADX_Regime_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
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
    
    # Align 1d ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Calculate 12h Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.ewm(span=20, adjust=False, min_periods=20).mean()
    dev = 2 * close_s.rolling(window=20, min_periods=20).std()
    upper_band = basis + dev
    lower_band = basis - dev
    bb_width = (upper_band - lower_band) / basis
    
    # Calculate 20th percentile of BB Width for squeeze detection (using expanding window)
    bb_width_series = pd.Series(bb_width.values)
    bb_width_percentile = bb_width_series.expanding(min_periods=50).quantile(0.20)
    squeeze_condition = bb_width < bb_width_percentile.values
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ema_20[i]) or
            np.isnan(squeeze_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            if squeeze_condition[i] and volume_confirm:
                # Breakout in direction of 1d ADX trend
                if adx_aligned[i] > 25:
                    # Strong trend: breakout in ADX direction
                    # Use 1d +DI/-DI for trend direction
                    plus_dm_14 = plus_dm.rolling(window=14, min_periods=14).sum()
                    minus_dm_14 = minus_dm.rolling(window=14, min_periods=14).sum()
                    plus_di_14 = 100 * (plus_dm_14 / atr)
                    minus_di_14 = 100 * (minus_dm_14 / atr)
                    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di_14.values)
                    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di_14.values)
                    
                    if plus_di_aligned[i] > minus_di_aligned[i]:
                        # Uptrend: long on break above upper band
                        if close[i] > upper_band[i]:
                            signals[i] = 0.25
                            position = 1
                    else:
                        # Downtrend: short on break below lower band
                        if close[i] < lower_band[i]:
                            signals[i] = -0.25
                            position = -1
                else:
                    # Weak trend: wait for stronger confirmation (price closes outside bands)
                    if close[i] > upper_band[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < lower_band[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: price returns to basis OR volatility expands (BB Width > 50th percentile) OR volume drops
            bb_width_percentile_50 = bb_width_series.expanding(min_periods=50).quantile(0.50).iloc[i] if i < len(bb_width_series) else 0.05
            if (close[i] <= basis[i] or 
                bb_width[i] > bb_width_percentile_50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to basis OR volatility expands OR volume drops
            bb_width_percentile_50 = bb_width_series.expanding(min_periods=50).quantile(0.50).iloc[i] if i < len(bb_width_series) else 0.05
            if (close[i] >= basis[i] or 
                bb_width[i] > bb_width_percentile_50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals