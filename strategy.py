#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA34 trend filter
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Long when price breaks above Camarilla R3 AND volume spike AND price > 1w EMA34
# Short when price breaks below Camarilla S3 AND volume spike AND price < 1w EMA34
# ATR-based stoploss: exit when price moves against position by 2.0 * ATR(14)
# Camarilla levels from 1d provide structure; volume confirms breakout strength; 1w EMA filters trend direction
# Works in bull via breakout longs above EMA, in bear via breakdown shorts below EMA

name = "12h_Camarilla_R3S3_1wEMA34_VolumeSpike_ATRStop_v1"
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
    
    # Calculate 1d OHLC for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # R3 = close + 1.1*(high-low)*1.1/4 = close + 1.1*(high-low)*0.275
    # S3 = close - 1.1*(high-low)*1.1/4 = close - 1.1*(high-low)*0.275
    # Actually: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # Simpler: range = high - low; R3 = close + range * 1.1/4; S3 = close - range * 1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    camarilla_R3 = prev_close + prev_range * 1.1 / 4.0
    camarilla_S3 = prev_close - prev_range * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Calculate 1w EMA(34) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma_30)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr0 = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 34)  # warmup for volume MA and 1w EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_30[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Camarilla R3 AND above 1w EMA34
                if (curr_close > curr_R3 and 
                    curr_close > curr_ema_34_1w):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla S3 AND below 1w EMA34
                elif (curr_close < curr_S3 and 
                      curr_close < curr_ema_34_1w):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # ATR-based stoploss: exit when price drops below entry - 2.0 * ATR
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # ATR-based stoploss: exit when price rises above entry + 2.0 * ATR
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals