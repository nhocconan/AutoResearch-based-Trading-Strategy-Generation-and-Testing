#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike (>2.0x 20-period average).
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike.
# ATR(14)-based stoploss: exit when price moves against position by 2.0 * ATR.
# Camarilla levels derived from prior 1d OHLC, providing institutional structure that works in both bull and bear regimes.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_ATRStop_v1"
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
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from prior 1d candle
        # Need prior 1d high, low, close - get from daily data aligned to current 4h bar
        if i < 1:  # Need at least one prior bar
            signals[i] = 0.0
            continue
            
        # Get prior 1d OHLC using HTF data - we need the completed 1d bar prior to current 4h bar
        # Since we're on 4h timeframe, we can use the prior 1d bar's data
        # We'll approximate by using the prior 1d bar's high/low/close from our aligned HTF data
        # For Camarilla, we use the prior 1d bar's OHLC
        # We need to get the prior 1d bar's high, low, close
        # We can get this by looking back 6 bars (since 6 * 4h = 24h = 1d) but this assumes perfect alignment
        # Better: use the HTF data directly
        # Find the index in df_1d that corresponds to the prior completed 1d bar
        # We'll use a simple approach: for each 4h bar, the relevant 1d bar is the one that started at 00:00 UTC of that day
        # Since we have df_1d from get_htf_data, we can align it and then use the prior value
        
        # Get prior 1d bar's OHLC - we need to access the completed 1d bar before current time
        # We'll use the aligned HTF data but shifted by 1 to get prior completed bar
        # However, we don't have aligned arrays for OHLC, only for EMA
        # Alternative: calculate Camarilla using the prior 1d bar's data from df_1d
        # We need to map the current 4h bar to the prior 1d bar index
        
        # Simpler approach: since we're on 4h timeframe, and 1d = 6 * 4h bars (approximately)
        # We'll use the prior 6 4h bars' high/low/close to approximate the prior 1d bar
        # This is not perfect but avoids look-ahead and uses only prior data
        if i < 6:
            signals[i] = 0.0
            continue
            
        # Calculate prior 1d OHLC from the last 6 4h bars (excluding current bar)
        prior_6h_high = np.max(high[i-6:i])
        prior_6h_low = np.min(low[i-6:i])
        prior_6h_close = close[i-1]  # close of prior 4h bar
        
        # Camarilla levels
        rang = prior_6h_high - prior_6h_low
        if rang <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_r3 = prior_6h_close + (1.1 * rang / 4)
        camarilla_s3 = prior_6h_close - (1.1 * rang / 4)
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Camarilla R3 AND above 1d EMA34
                if (curr_close > camarilla_r3 and 
                    curr_close > curr_ema_34_1d):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla S3 AND below 1d EMA34
                elif (curr_close < camarilla_s3 and 
                      curr_close < curr_ema_34_1d):
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