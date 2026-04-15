#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume spike and 1d ADX trend filter
# Camarilla levels (L3, L4, H3, H4) act as intraday support/resistance.
# Enter long at L3/L4 with bullish engulfing + volume spike.
# Enter short at H3/H4 with bearish engulfing + volume spike.
# Use 1d ADX > 25 to ensure trading only in trending markets, avoiding chop.
# Works in bull (buy dips at support) and bear (sell rallies at resistance).
# Conservative sizing (0.25) to limit trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14 = adx  # already smoothed
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # Calculate for previous day (shift by 1)
    typical_price_prev = np.concatenate([[np.nan], typical_price[:-1]])
    range_prev = np.concatenate([[np.nan], range_1d[:-1]])
    
    # Camarilla levels
    L3 = typical_price_prev - 1.1 * range_prev / 6
    L4 = typical_price_prev - 1.1 * range_prev / 4
    H3 = typical_price_prev + 1.1 * range_prev / 4
    H4 = typical_price_prev + 1.1 * range_prev / 6
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_) & (open_ < close_) & (close > open_.shift(1)) & (open_ < close_.shift(1))
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close < open_) & (open_ > close_) & (close < open_.shift(1)) & (open_ > close_.shift(1))
    # Need to handle arrays properly
    open_ = prices['open'].values
    bullish_engulf = (close > open_) & (open_ < close_) & (np.roll(close, 1) < np.roll(open_, 1)) & (np.roll(open_, 1) > np.roll(close, 1))
    bearish_engulf = (close < open_) & (open_ > close_) & (np.roll(close, 1) > np.roll(open_, 1)) & (np.roll(open_, 1) < np.roll(close, 1))
    # First bar: no engulfing
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(L3[i]) or np.isnan(L4[i]) or np.isnan(H3[i]) or np.isnan(H4[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        if adx_14_aligned[i] <= 25:
            # Hold previous position or stay flat
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Long: price at L3/L4 + bullish engulfing + volume spike
        if ((close[i] <= L3[i] * 1.005 or close[i] <= L4[i] * 1.005) and
            bullish_engulf[i] and
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price at H3/H4 + bearish engulfing + volume spike
        elif ((close[i] >= H3[i] * 0.995 or close[i] >= H4[i] * 0.995) and
              bearish_engulf[i] and
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal or price moves significantly away from level
        elif i > 0:
            prev_signal = signals[i-1]
            if prev_signal == 0.25 and (close[i] >= L4[i] * 1.01 or bearish_engulf[i]):
                signals[i] = 0.0
            elif prev_signal == -0.25 and (close[i] <= H4[i] * 0.99 or bullish_engulf[i]):
                signals[i] = 0.0
            else:
                signals[i] = prev_signal
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_CamarillaReversal_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0