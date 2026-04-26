#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX momentum with volume confirmation and choppiness regime filter on 4h timeframe.
TRIX (12,20) filters noise and identifies sustainable momentum. Volume spike ensures institutional participation.
Choppiness regime (CHOP > 61.8) enables mean-reversion logic in ranging markets, while (CHOP < 38.2) enables trend-following.
Works in bull/bear markets by adapting to regime: trend-follow in strong trends, mean-revert in chop.
Targets 75-200 total trades over 4 years via disciplined entry requiring TRIX crossover, volume confirmation, and regime alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter (regime proxy)
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate TRIX(12,20) on close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then ROC of 20 periods
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3.pct_change(periods=20))
    trix = trix_raw.values
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: volume > 1.8x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.8 * vol_median_20)
    
    # Choppiness Index on 4h (primary timeframe)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (highest_high - lowest_low))) / log10(n)
    # where n = 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * (np.log10(atr_sum / range_hl) / np.log10(14))
    
    # Reduced fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 12+12+12+20=56 for TRIX, 14 for ATR/CHOP, 20 for volume median
    start_idx = max(56, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or
            np.isnan(atr[i]) or np.isnan(chop[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        trix_val = trix[i]
        trix_signal_val = trix_signal[i]
        atr_val = atr[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        ema_34_val = ema_34_1d_aligned[i]
        
        if position == 0:
            # Flat - look for entry
            # Regime logic: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
            if chop_val > 61.8:
                # Ranging market: mean reversion at extremes
                # Long: TRIX crosses above signal from below AND volume spike AND price below 1d EMA34 (oversold in downtrend)
                long_entry = (trix_val > trix_signal_val and 
                             trix[i-1] <= trix_signal[i-1] and  # crossed up
                             vol_spike and 
                             close_val < ema_34_val)
                # Short: TRIX crosses below signal from above AND volume spike AND price above 1d EMA34 (overbought in uptrend)
                short_entry = (trix_val < trix_signal_val and 
                              trix[i-1] >= trix_signal[i-1] and  # crossed down
                              vol_spike and 
                              close_val > ema_34_val)
            else:
                # Trending market: trend following
                # Long: TRIX crosses above signal from below AND volume spike AND price above 1d EMA34
                long_entry = (trix_val > trix_signal_val and 
                             trix[i-1] <= trix_signal[i-1] and  # crossed up
                             vol_spike and 
                             close_val > ema_34_val)
                # Short: TRIX crosses below signal from above AND volume spike AND price below 1d EMA34
                short_entry = (trix_val < trix_signal_val and 
                              trix[i-1] >= trix_signal[i-1] and  # crossed down
                              vol_spike and 
                              close_val < ema_34_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on TRIX cross down, ATR stoploss, or regime change to extreme chop
            stop_price = entry_price - 2.5 * atr_val
            if (trix_val < trix_signal_val and trix[i-1] >= trix_signal[i-1]) or \
               close_val < stop_price or \
               chop_val > 70.0:  # extreme chop - exit mean reversion
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on TRIX cross up, ATR stoploss, or regime change to extreme chop
            stop_price = entry_price + 2.5 * atr_val
            if (trix_val > trix_signal_val and trix[i-1] <= trix_signal[i-1]) or \
               close_val > stop_price or \
               chop_val > 70.0:  # extreme chop - exit mean reversion
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0