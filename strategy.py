#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TrixMomentum_VolumeRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d TRIX (TRIPLE EXPONENTIAL MOVING AVERAGE)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix_raw.fillna(0).values
    
    # Align 1d TRIX to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume filter: current 4h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Choppiness regime filter (1d) - avoids choppy markets
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    
    # ATR(14)
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(ATR14)/ (HH - LL)) / log10(14)
    sum_atr = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    chop = np.where(hh_ll > 0, 100 * np.log10(sum_atr / hh_ll) / np.log10(14), 50)
    chop = np.nan_to_num(chop, nan=50)
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Warmup for TRIX and volume
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_aligned[i]
        chop_val = chop_aligned[i]
        vol_filter = volume_filter[i]
        
        # Only trade in trending markets (Chop < 61.8) or ranging markets (Chop > 61.8) with momentum
        if position == 0:
            # Enter long: TRIX > 0 (bullish momentum) + volume filter + not extreme chop
            if trix_val > 0 and vol_filter and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX < 0 (bearish momentum) + volume filter + not extreme chop
            elif trix_val < 0 and vol_filter and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns negative or chop becomes too high (range)
            if trix_val < 0 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns positive or chop becomes too high (range)
            if trix_val > 0 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

if __name__ == "__main__":
    # Quick self-test
    import yfinance as yf
    data = yf.download("BTC-USD", start="2021-01-01", end="2024-01-01", interval="1h")
    data.reset_index(inplace=True)
    data.rename(columns={"Datetime": "open_time"}, inplace=True)
    # For testing only - real implementation uses 4h data from Binance
    print("Self-test placeholder - actual testing uses 4h Binance data")