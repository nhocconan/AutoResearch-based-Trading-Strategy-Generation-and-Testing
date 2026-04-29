#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Long when Bollinger Bands squeeze (BW < 20th percentile) breaks above upper band AND 1d EMA50 uptrend
# Short when Bollinger Bands squeeze breaks below lower band AND 1d EMA50 downtrend
# Volume confirmation reduces false breaks. Works in both bull/bear by following 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_BollingerSqueeze_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2.0) on 6h data
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Bollinger Band Width (BW) for squeeze detection
    bb_width = (upper_band - lower_band) / sma
    # Squeeze: BW below 20th percentile of its 50-period lookback
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    is_squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 20)  # warmup for BB, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_sma = sma[i]
        curr_ema50 = ema50_aligned[i]
        curr_is_squeeze = is_squeeze[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade during Bollinger Band squeeze with volume confirmation
            if curr_is_squeeze and curr_volume_confirm:
                # Bullish breakout: price breaks above upper band AND bullish regime
                if curr_close > curr_upper and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower band AND bearish regime
                elif curr_close < curr_lower and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to middle band (SMA) OR regime changes to bearish
            if curr_close < curr_sma or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to middle band (SMA) OR regime changes to bullish
            if curr_close > curr_sma or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals