#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for key indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20
    
    # Daily Bollinger Band Percentile (252-day lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Daily RSI (14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi[np.isnan(rsi)] = 50  # neutral for warmup
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Daily volume ratio (current / 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 4-hour price data for entry timing (using 4h for precision)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (20-period)
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bb_percentile = bb_width_percentile_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        upper_bband = donchian_upper_aligned[i]
        lower_bband = donchian_lower_aligned[i]
        
        # Regime filter: low volatility environment (Bollinger Band width in lowest 30%)
        low_vol_regime = bb_percentile < 30
        
        # Mean reentry conditions with volume confirmation
        if position == 0:
            # Long setup: price at lower Donchian band + oversold RSI + volume spike + low vol regime
            if (price_close <= lower_bband * 1.001 and  # slight buffer for entry
                rsi_val < 30 and 
                vol_ratio_val > 1.8 and 
                low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short setup: price at upper Donchian band + overbought RSI + volume spike + low vol regime
            elif (price_close >= upper_bband * 0.999 and  # slight buffer for entry
                  rsi_val > 70 and 
                  vol_ratio_val > 1.8 and 
                  low_vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: mean reversion or volatility expansion
            if position == 1:
                # Exit long: price reaches midpoint OR RSI overbought OR volatility expands
                midpoint = (upper_bband + lower_bband) / 2
                if (price_close >= midpoint or 
                    rsi_val > 60 or 
                    bb_percentile > 70):  # volatility expansion
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # hold
            elif position == -1:
                # Exit short: price reaches midpoint OR RSI oversold OR volatility expands
                midpoint = (upper_bband + lower_bband) / 2
                if (price_close <= midpoint or 
                    rsi_val < 40 or 
                    bb_percentile > 70):  # volatility expansion
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # hold
    
    return signals

name = "4h_BollingerBand_Width_Percentile_Donchian_MeanReversion"
timeframe = "4h"
leverage = 1.0