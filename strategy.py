#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_TRIX_VolumeSpike_CHR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h: TRIX (12-period EMA triple smoothed) ===
    close_12h = df_12h['close'].values
    close_series = pd.Series(close_12h)
    # TRIX = EMA(EMA(EMA(close), 12), 12), 12) then percent change
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3.pct_change())
    trix = trix_raw.values
    
    # === 12h: Volume spike detection ===
    volume_12h = df_12h['volume'].values
    vol_series = pd.Series(volume_12h)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = (vol_series / vol_ma).values
    
    # Align TRIX and volume ratio to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    
    # === 4h: Choppiness Index (CHOP) regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TRUE RANGE over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(sumTR / (HH - LL)) / log10(14)
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl > 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # === 4h: Volume spike for confirmation ===
    volume = prices['volume'].values
    vol_series_4h = pd.Series(volume)
    vol_ma_4h = vol_series_4h.rolling(window=20, min_periods=20).mean()
    vol_ratio_4h = (vol_series_4h / vol_ma_4h).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        trix_val = trix_aligned[i]
        vol_ratio_12h = vol_ratio_aligned[i]
        current_chop = chop[i]
        current_atr = atr[i]
        current_close = close[i]
        current_vol_ratio = vol_ratio_4h[i]
        
        # Skip if any value is NaN
        if (np.isnan(trix_val) or np.isnan(vol_ratio_12h) or 
            np.isnan(current_chop) or np.isnan(current_atr) or 
            np.isnan(current_vol_ratio)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Conditions ===
        # TRIX momentum: positive = bullish, negative = bearish
        # Volume spike: > 1.8x average
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending (follow momentum)
        vol_spike_12h = vol_ratio_12h > 1.8
        vol_spike_4h = current_vol_ratio > 1.8
        
        if position == 0:
            # Long: TRIX turning up + volume spike + trending market (CHOP < 38.2)
            if trix_val > 0 and trix_val > trix_aligned[i-1] and vol_spike_12h and vol_spike_4h and current_chop < 38.2:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: TRIX turning down + volume spike + trending market (CHOP < 38.2)
            elif trix_val < 0 and trix_val < trix_aligned[i-1] and vol_spike_12h and vol_spike_4h and current_chop < 38.2:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
            
            # Mean reversion in ranging markets: fade extremes
            elif current_chop > 61.8:
                # Buy near low, sell near high in range
                if current_close <= lowest_low[i] + 0.1 * range_hl[i]:  # Near low
                    signals[i] = 0.20
                    position = 1
                    entry_price = current_close
                elif current_close >= highest_high[i] - 0.1 * range_hl[i]:  # Near high
                    signals[i] = -0.20
                    position = -1
                    entry_price = current_close
        
        elif position == 1:
            # Long exit: TRIX turns down OR stop loss OR range high
            if trix_val < 0 and trix_val < trix_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            elif current_chop > 61.8 and current_close >= highest_high[i] - 0.1 * range_hl[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns up OR stop loss OR range low
            if trix_val > 0 and trix_val > trix_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            elif current_chop > 61.8 and current_close <= lowest_low[i] + 0.1 * range_hl[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals