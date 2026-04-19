#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Choppiness_Regime_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ATR(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index on 1d (14-period)
    # Chop = 100 * log10(sum(atr over n) / (log10(highest high - lowest low) * n))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    chop = 100 * np.log10(atr_sum / (range_14 * 14))
    chop = np.where(range_14 == 0, 100, chop)  # Avoid division by zero
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike on 12h (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        chop_val = chop_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_confirm = volume_spike[i]
        
        # Regime filter: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (trend follow)
        # In ranging markets: fade moves (sell highs, buy lows)
        # In trending markets: follow trend (buy highs, sell lows)
        if chop_val > 61.8:  # Ranging market
            # Mean reversion: sell when price near upper band, buy when near lower band
            # Use price position relative to recent range
            if i >= 20:
                recent_high = np.max(high[i-20:i+1])
                recent_low = np.min(low[i-20:i+1])
                if recent_high > recent_low:
                    price_position = (close[i] - recent_low) / (recent_high - recent_low)
                    # Sell when price in upper 30% of range, buy when in lower 30%
                    if price_position > 0.7 and vol_confirm:
                        if position <= 0:
                            signals[i] = -0.25
                            position = -1
                    elif price_position < 0.3 and vol_confirm:
                        if position >= 0:
                            signals[i] = 0.25
                            position = 1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
                
        else:  # Trending market (Chop < 61.8)
            # Follow weekly trend: long when price > weekly EMA, short when price < weekly EMA
            # But only enter on pullbacks to reduce whipsaw
            if i >= 10:
                # Calculate short-term pullback
                recent_high_10 = np.max(high[i-10:i+1])
                recent_low_10 = np.min(low[i-10:i+1])
                
                if ema_trend > close[i]:  # Weekly uptrend
                    # Look for pullback to buy (price near recent low)
                    if close[i] < recent_low_10 + 0.3 * (recent_high_10 - recent_low_10):
                        if vol_confirm:
                            if position <= 0:
                                signals[i] = 0.25
                                position = 1
                else:  # Weekly downtrend
                    # Look for pullback to sell (price near recent high)
                    if close[i] > recent_high_10 - 0.3 * (recent_high_10 - recent_low_10):
                        if vol_confirm:
                            if position >= 0:
                                signals[i] = -0.25
                                position = -1
            else:
                signals[i] = 0.0
        
        # Exit conditions: reverse signal or volatility expansion
        if position == 1:
            # Exit long: chop increases significantly (trend weakening) or opposite signal
            if chop_val > 70 or (chop_val > 61.8 and close[i] > np.max(high[i-5:i+1]) * 0.995):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: chop increases significantly or opposite signal
            if chop_val > 70 or (chop_val > 61.8 and close[i] < np.min(low[i-5:i+1]) * 1.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals