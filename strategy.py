#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) with 4h ADX(14) trend filter and 1d Bollinger Band regime
# Long when RSI(14) < 30 + price > SMA(200) + ADX(14) > 25 (trending) + 1d BB width > 50th percentile (volatile)
# Short when RSI(14) > 70 + price < SMA(200) + ADX(14) > 25 (trending) + 1d BB width > 50th percentile
# Exit when RSI crosses 50 (mean reversion) or BB width < 30th percentile (low volatility)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_rsi14_4h_adx_1d_bb_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 1d data for Bollinger Bands and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h ADX(14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = np.diff(low_4h, prepend=low_4h[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    bb_mid = close_1d_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width for regime filter
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Align BB levels to 1h
    bb_mid_aligned = align_htf_to_ltf(prices, df_1d, bb_mid)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h SMA(200)
    close_s = pd.Series(close)
    sma_200 = close_s.rolling(window=200, min_periods=200).mean().values
    
    # 1h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(sma_200[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses 50 (mean reversion) or low volatility regime
            elif rsi[i] >= 50 or bb_width_percentile_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses 50 (mean reversion) or low volatility regime
            elif rsi[i] <= 50 or bb_width_percentile_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extremes with trend and volatility filters
            # Trend filter: ADX > 25 (trending market)
            trending = adx_aligned[i] > 25
            # Volatility filter: BB width > 50th percentile (volatile enough)
            volatile_regime = bb_width_percentile_aligned[i] > 50
            
            # Long: RSI oversold + price above SMA200 + trending + volatile regime
            if rsi[i] < 30 and close[i] > sma_200[i] and trending and volatile_regime:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI overbought + price below SMA200 + trending + volatile regime
            elif rsi[i] > 70 and close[i] < sma_200[i] and trending and volatile_regime:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals