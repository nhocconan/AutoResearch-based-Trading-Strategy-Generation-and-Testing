#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_trix_volume_chop_v1
# Uses TRIX (12-period) on 4h for momentum, with volume confirmation (>1.5x 20-period average).
# Adds a chop regime filter using 4h CHOP(14): only trade when CHOP < 50 (trending) or CHOP > 61.8 (range) for mean reversion.
# Long when TRIX crosses above 0 AND volume confirms AND (CHOP < 50 OR close < lower Bollinger Band(20,2)).
# Short when TRIX crosses below 0 AND volume confirms AND (CHOP < 50 OR close > upper Bollinger Band(20,2)).
# Exits when TRIX returns to zero.
# Designed for 4h timeframe with daily trend filter: only take long if 1d EMA(50) > 1d EMA(200), short if opposite.
# Target: 20-40 trades/year to minimize fee drag, works in both bull (trend follow) and bear (mean reversion in range).

name = "4h_1d_trix_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h TRIX (12-period)
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then / previous value * 100
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.fillna(0).values
    
    # 4h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # 4h Bollinger Bands (20,2) for mean reversion signals in ranging markets
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = (bb_mid + 2 * bb_std).values
    bb_lower = (bb_mid - 2 * bb_std).values
    
    # 4h Choppiness Index (CHOP) - measures if market is choppy (range) or trending
    # CHOP = 100 * log10(sum(ATR(1), 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low[1:] - close[:-1]))
    tr1 = np.insert(tr1, 0, high[0] - low[0])  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr1.rolling(window=14, min_periods=14).sum() / (max_high - min_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(trix[i]):
            signals[i] = 0.0
            continue
        
        # Determine market regime and direction bias
        is_trending = chop[i] < 50
        is_ranging = chop[i] > 61.8
        daily_uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        daily_downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        # Long conditions
        long_signal = False
        if is_trending and daily_uptrend:
            # In trending + daily uptrend: follow TRIX cross above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                long_signal = True
        elif is_ranging:
            # In ranging: mean reversion from lower Bollinger Band
            if close[i] <= bb_lower[i] and close[i-1] > bb_lower[i-1]:
                long_signal = True
        
        # Short conditions
        short_signal = False
        if is_trending and daily_downtrend:
            # In trending + daily downtrend: follow TRIX cross below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                short_signal = True
        elif is_ranging:
            # In ranging: mean reversion from upper Bollinger Band
            if close[i] >= bb_upper[i] and close[i-1] < bb_upper[i-1]:
                short_signal = True
        
        # Execute signals with volume confirmation
        if long_signal and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit when TRIX crosses zero (mean reversion to momentum equilibrium)
        elif position == 1 and trix[i] < 0 and trix[i-1] >= 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix[i] > 0 and trix[i-1] <= 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals