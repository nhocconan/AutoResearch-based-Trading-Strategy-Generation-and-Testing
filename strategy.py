#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivots provide strong support/resistance levels; breakouts with volume confirm institutional interest.
# EMA34 on 1d ensures we trade with the higher timeframe trend to avoid counter-trend whipsaws.
# Long: price breaks above R3 (1.118*close - 0.118*low) AND close > EMA34_1d AND volume spike
# Short: price breaks below S3 (1.118*high - 0.118*close) AND close < EMA34_1d AND volume spike
# ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14) to allow for volatility
# Discrete sizing 0.25 to control risk and minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull via breakout longs, in bear via breakout shorts during rallies, with trend filter avoiding false signals.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "12h"
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
    
    # Calculate Camarilla levels (R3, S3) from previous bar
    # R3 = close + 1.118*(high - low)
    # S3 = close - 1.118*(high - low)
    camarilla_range = high - low
    r3 = close + 1.118 * camarilla_range
    s3 = close - 1.118 * camarilla_range
    # Shift by 1 to use previous bar's levels (no look-ahead)
    r3_prev = np.concatenate([[np.nan], r3[:-1]])
    s3_prev = np.concatenate([[np.nan], s3[:-1]])
    
    # Calculate 1d EMA(34) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34, 14)  # warmup for volume MA, EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r3_prev[i]) or np.isnan(s3_prev[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_prev[i]
        curr_s3 = s3_prev[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above R3 AND close > EMA34 (bullish bias)
                if (curr_high > curr_r3 and 
                    curr_close > curr_ema_34_1d):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 AND close < EMA34 (bearish bias)
                elif (curr_low < curr_s3 and 
                      curr_close < curr_ema_34_1d):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # ATR-based stoploss: exit when price drops below entry - 2.5 * ATR
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # ATR-based stoploss: exit when price rises above entry + 2.5 * ATR
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals