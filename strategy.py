#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1w trend filter and volume confirmation
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 80-120 total trades over 4 years (20-30/year).
# Elder Ray measures bull/bear power via EMA13. Long when Bear Power < 0 and Bull Power rising.
# Short when Bull Power > 0 and Bear Power falling. 1w EMA34 filters for major trend alignment.
# Volume spike ensures institutional participation. Avoids choppy markets via ADX(14) > 20.
# Works in bull via trend-following longs, in bear via selective shorts on rallies.

name = "6h_ElderRay_1wEMA34_VolumeSpike_ADXFilter_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # Calculate 1w EMA(34) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    # ADX(14) for regime filter - avoid choppy markets
    # +DI = 100 * EMA(14 of +DM) / ATR
    # -DI = 100 * EMA(14 of -DM) / ATR
    # ADX = EMA(14 of |+DI - -DI| / (+DI + -DI))
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
    adx_filter = adx > 20  # Trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(13, 30, 34, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_30[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_adx_filter = adx_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market
            if curr_volume_spike and curr_adx_filter:
                # Bullish entry: Bear Power < 0 (below EMA) AND Bull Power rising (improving momentum)
                if (curr_bear_power < 0 and 
                    curr_bull_power > bull_power[i-1] and  # Bull Power increasing
                    curr_close > curr_ema_34_1w):  # Above weekly EMA34 (bullish bias)
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Bull Power > 0 (above EMA) AND Bear Power falling (deteriorating momentum)
                elif (curr_bull_power > 0 and 
                      curr_bear_power < bear_power[i-1] and  # Bear Power decreasing (more negative)
                      curr_close < curr_ema_34_1w):  # Below weekly EMA34 (bearish bias)
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative OR loses weekly trend OR Bear Power rises sharply
            if (curr_bull_power < 0 or  # Lost bullish momentum
                curr_close < curr_ema_34_1w or  # Lost weekly uptrend
                curr_bear_power > bear_power[i-1] * 0.5):  # Bear Power recovering (exit long)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR loses weekly trend OR Bull Power rises sharply
            if (curr_bear_power > 0 or  # Lost bearish momentum
                curr_close > curr_ema_34_1w or  # Lost weekly downtrend
                curr_bull_power > bull_power[i-1] * 0.5):  # Bull Power recovering (exit short)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals