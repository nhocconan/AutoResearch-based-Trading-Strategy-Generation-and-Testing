#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Long when %R < -80 and rising from extreme.
# Short when %R > -20 and falling from extreme. 1d EMA50 filters for major trend alignment.
# Volume spike confirms participation. Avoids choppy markets via ADX(14) > 25.
# Designed for mean reversion in ranging markets and selective trend-following in strong trends.
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.25.

name = "4h_WilliamsR_1dEMA50_VolumeSpike_ADXFilter_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d EMA(50) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ADX(14) for regime filter - avoid choppy markets
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_filter = adx > 25  # Strongly trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(14, 20, 50, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_adx_filter = adx_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Mean reversion long: oversold (%R < -80) and rising from extreme
                if (curr_williams_r < -80 and 
                    curr_williams_r > williams_r[i-1] and  # Williams %R rising
                    curr_close > curr_ema_50_1d):  # Above daily EMA50 (bullish bias)
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Mean reversion short: overbought (%R > -20) and falling from extreme
                elif (curr_williams_r > -20 and 
                      curr_williams_r < williams_r[i-1] and  # Williams %R falling
                      curr_close < curr_ema_50_1d):  # Below daily EMA50 (bearish bias)
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                # Trend-following long: strong uptrend with pullback to oversold
                elif (curr_adx_filter and 
                      curr_close > curr_ema_50_1d and  # Above daily EMA50 (uptrend)
                      curr_williams_r < -80 and  # Oversold pullback
                      curr_williams_r > williams_r[i-1]):  # Williams %R rising from low
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Trend-following short: strong downtrend with pullback to overbought
                elif (curr_adx_filter and 
                      curr_close < curr_ema_50_1d and  # Below daily EMA50 (downtrend)
                      curr_williams_r > -20 and  # Overbought pullback
                      curr_williams_r < williams_r[i-1]):  # Williams %R falling from high
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Williams %R overbought OR breaks trend OR loses momentum
            if (curr_williams_r > -20 or  # Overbought - time to take profit
                curr_close < curr_ema_50_1d * 0.98 or  # Breaks below daily EMA50 (trend change)
                curr_williams_r < williams_r[i-1] * 0.9):  # Losing upward momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R oversold OR breaks trend OR loses momentum
            if (curr_williams_r < -80 or  # Oversold - time to take profit
                curr_close > curr_ema_50_1d * 1.02 or  # Breaks above daily EMA50 (trend change)
                curr_williams_r > williams_r[i-1] * 0.9):  # Losing downward momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals